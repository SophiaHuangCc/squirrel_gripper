import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from dgdm.data import UnconditionalDesignDataset
from dgdm.sampler import DGDMDesignDiffusion
from generator.dataloader import DesignBounds
from generator.diffusion_utils import ConditionalUnet1D


def main():
    p = argparse.ArgumentParser(description="Train DGDM unconditional valid-design prior")
    p.add_argument("--data_dir", required=True); p.add_argument("--save_dir", default="dgdm/runs/prior")
    p.add_argument("--epochs", type=int, default=500); p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=100); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda"); args = p.parse_args()
    torch.manual_seed(args.seed); device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    bounds = DesignBounds.defaults(); data = UnconditionalDesignDataset(args.data_dir, bounds)
    loader = DataLoader(data, args.batch_size, shuffle=True, num_workers=args.workers)
    net = ConditionalUnet1D(1, 0, [128, 256], 32)
    scheduler = DDIMScheduler(num_train_timesteps=args.timesteps, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    model = DGDMDesignDiffusion(net, scheduler, bounds).to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True); bounds.save(os.path.join(args.save_dir, "design_bounds.npz"))
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad(set_to_none=True)
            loss = model.training_loss(batch); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); model.ema.step(net)
            total += loss.item()
        print(f"epoch={epoch} loss={total/max(len(loader),1):.6f}")
        state = {"model": model.state_dict(), "ema": model.ema.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(state, os.path.join(args.save_dir, "last.pt"))

if __name__ == "__main__": main()
