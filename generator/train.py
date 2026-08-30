import argparse
import os
import sys
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, ".."))

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from generator.dataloader import DesignBounds, SquirrelDiffusionDataset
from generator.diffusion import SquirrelDesignDiffusion
from generator.diffusion_utils import ConditionalUnet1D


def parse_args():
    parser = argparse.ArgumentParser(description="Train squirrel finger diffusion generator.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="generator/runs/diffusion")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_timesteps", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--ema_power", type=float, default=0.75)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=25)
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device).float() for k, v in batch.items() if torch.is_tensor(v)}
            losses.append(model.training_loss(batch).item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def save_checkpoint(path, model, optimizer, epoch, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "noise_pred_net": model.noise_pred_net.state_dict(),
            "ema": model.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    bounds = DesignBounds.defaults()
    bounds_path = os.path.join(args.save_dir, "design_bounds.npz")
    bounds.save(bounds_path)

    dataset = SquirrelDiffusionDataset(
        dataset_dir=args.data_dir,
        bounds=bounds,
    )
    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    unet = ConditionalUnet1D(
        input_dim=1,
        global_cond_dim=8,
        down_dims=[128, 256],
        diffusion_step_embed_dim=32,
    )
    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    model = SquirrelDesignDiffusion(
        noise_pred_net=unet,
        noise_scheduler=scheduler,
        bounds=bounds,
        learning_rate=args.learning_rate,
        ema_power=args.ema_power,
        num_inference_steps=args.num_inference_steps,
    ).to(device)
    optimizer = model.optimizer()

    best_val = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.num_epochs}")
        for batch in progress:
            batch = {k: v.to(device).float() for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()
            running.append(loss.item())
            progress.set_postfix(train_loss=f"{sum(running) / len(running):.5f}")

        val_loss = evaluate(model, val_loader, device)
        train_loss = float(sum(running) / max(len(running), 1))
        print(f"[DIFFUSION] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(args.save_dir, "best.pt"), model, optimizer, epoch, args)

        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            save_checkpoint(os.path.join(args.save_dir, f"epoch_{epoch:04d}.pt"), model, optimizer, epoch, args)

    save_checkpoint(os.path.join(args.save_dir, "last.pt"), model, optimizer, args.num_epochs, args)
    print("[SAVED]", os.path.abspath(args.save_dir))


if __name__ == "__main__":
    main()
