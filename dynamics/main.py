import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dynamics.dataloader import DynamicsDataset
from dynamics.trainer import Trainer
from dynamics.parser import parse


def validate(args, val_loader, trainer):
    print('--- Validation Step ---')
    total_val_loss = 0.0
    total_mae = 0.0
    total_mae_per_dim = None

    trainer.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            target = batch['target_metrics'].to(trainer.device).float()
            task_params = batch['task_params'].to(trainer.device).float()
            design_params = batch['design_params'].to(trainer.device).float()
            init_config = batch['init_config'].to(trainer.device).float()

            pred, loss = trainer.inference(task_params, design_params, init_config, target)

            batch_mae = torch.abs(pred - target)   # shape [B, 2]

            total_val_loss += loss.item()
            total_mae += batch_mae.mean().item()

            if total_mae_per_dim is None:
                total_mae_per_dim = batch_mae.mean(dim=0)
            else:
                total_mae_per_dim += batch_mae.mean(dim=0)

    avg_loss = total_val_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    avg_mae_per_dim = total_mae_per_dim / len(val_loader)

    print(
        f'Val Loss: {avg_loss:.4f} | '
        f'MAE total: {avg_mae:.4f} | '
        f'contacts: {avg_mae_per_dim[0].item():.4f}, '
        f'disturbance_score: {avg_mae_per_dim[1].item():.4f}'
    )

    return avg_loss, avg_mae, avg_mae_per_dim

def train(args):
    wandb.init(
        project='squirrel-gripper-dynamics',
        config=args,
        dir=args.save_dir,
        name=args.wandb_id,
    )

    train_dataset = DynamicsDataset(dataset_dir=args.data_dir)
    val_dataset = DynamicsDataset(dataset_dir=args.test_data_dir)

    sample = train_dataset[0]
    print("\n[DATA CHECK]")
    print("task_params:", sample["task_params"])
    print("design_params:", sample["design_params"])
    print("init_config:", sample["init_config"])
    print("target_metrics:", sample["target_metrics"])
    print("target shape:", sample["target_metrics"].shape)
    print("=" * 50)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    trainer = Trainer(args)
    trainer.create_model()

    if args.mode == 'validate':
        validate(args, val_loader, trainer)
        return

    if args.mode == 'train':
        best_val_loss = float('inf')
        last_best_epoch = 0
        
        for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
            average_loss = 0

            for idx_batch, batch in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
                target = batch['target_metrics'].to(trainer.device).float()
                task_params = batch['task_params'].to(trainer.device).float()
                design_params = batch['design_params'].to(trainer.device).float()
                init_config = batch['init_config'].to(trainer.device).float()

                loss, pred = trainer.step(
                    target,
                    task_params,
                    design_params,
                    init_config,
                )

                average_loss += loss

                if idx_batch % args.save_ckpt_step == 0:
                    trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pt'))

            trainer.lr_scheduler.step()
            
            # Logging
            if epoch % args.val_step == 0:

                val_loss, avg_mae, avg_mae_per_dim = validate(args, val_loader, trainer)

                wandb.log({
                    'train/loss': average_loss / len(train_loader),
                    'val/loss': val_loss,
                    'val/avg_mae_total': avg_mae,
                    'val/mae_contacts': avg_mae_per_dim[0].item(),
                    'val/mae_disturbance_score': avg_mae_per_dim[1].item(),
                })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    trainer.save_checkpoint(os.path.join(args.save_dir, 'best.pt'))
                    last_best_epoch = epoch
                elif epoch - last_best_epoch >= args.patience:
                    print('Early stopping triggered.')
                    break
    wandb.finish()

if __name__ == '__main__':
    args = parse()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)