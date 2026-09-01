import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader, random_split
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from dgdm.data import InteractionProfileDataset, PROFILE_CHANNELS
from dgdm.models import InteractionProfileModel, masked_profile_loss


def main():
    p = argparse.ArgumentParser(description="Train task-agnostic DGDM interaction dynamics")
    p.add_argument("--data_dir", required=True); p.add_argument("--save_dir", default="dgdm/runs/dynamics")
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--profile_steps", type=int, default=32); p.add_argument("--val_ratio", type=float, default=.1)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda")
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--timesteps_per_batch", type=int, default=4)
    args = p.parse_args()
    torch.manual_seed(args.seed); device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = InteractionProfileDataset(args.data_dir, args.profile_steps)
    nval = max(1, round(len(dataset)*args.val_ratio)); train, val = random_split(dataset, [len(dataset)-nval, nval], generator=torch.Generator().manual_seed(args.seed))
    loaders = {"train": DataLoader(train,args.batch_size,shuffle=True,num_workers=args.workers), "val": DataLoader(val,args.batch_size,num_workers=args.workers)}
    model = InteractionProfileModel(profile_steps=args.profile_steps, channels=len(PROFILE_CHANNELS)).to(device)
    scheduler = DDIMScheduler(
        num_train_timesteps=args.timesteps, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); os.makedirs(args.save_dir, exist_ok=True); best=float("inf")
    for epoch in range(1,args.epochs+1):
        metrics={}
        for split, loader in loaders.items():
            model.train(split=="train"); total=0.0
            for b in loader:
                b={k:v.to(device) for k,v in b.items()}
                with torch.set_grad_enabled(split=="train"):
                    repeats = args.timesteps_per_batch if split == "train" else 1
                    clean = b["design_unit"].repeat(repeats, 1)
                    scenario = b["scenario"].repeat(repeats, 1)
                    profile = b["profile"].repeat(repeats, 1, 1)
                    mask = b["profile_mask"].repeat(repeats, 1, 1)
                    if split == "train":
                        timestep = torch.randint(0, args.timesteps, (clean.shape[0],), device=device)
                        noisy = scheduler.add_noise(clean, torch.randn_like(clean), timestep)
                    else:
                        timestep = torch.zeros(clean.shape[0], dtype=torch.long, device=device)
                        noisy = clean
                    tau = timestep.float() / float(args.timesteps)
                    loss=masked_profile_loss(model(noisy,scenario,tau),profile,mask)
                    if split=="train": opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                total += loss.item()
            metrics[split]=total/max(len(loader),1)
        print(f"epoch={epoch} train={metrics['train']:.6f} val={metrics['val']:.6f}")
        state={"model":model.state_dict(),"epoch":epoch,"profile_steps":args.profile_steps,"profile_channels":PROFILE_CHANNELS,"noise_conditioned":True,"design_coordinates":"diffusion_unit","num_train_timesteps":args.timesteps,"args":vars(args)}
        torch.save(state,os.path.join(args.save_dir,"last.pt"))
        if metrics["val"] < best: best=metrics["val"]; torch.save(state,os.path.join(args.save_dir,"best.pt"))

if __name__ == "__main__": main()
