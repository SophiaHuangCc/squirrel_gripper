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
import numpy as np
import matplotlib.pyplot as plt
import datetime

from dynamics.dataloader import DynamicsDataset
from dynamics.trainer import Trainer
from dynamics.parser import parse


def evaluate_prediction_quality(trainer, dataloader, args, epoch, split_name="val"):
    trainer.model.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            pred, _ = trainer.inference(
                task_params=batch["task_params"].to(trainer.device).float(),
                design_params=batch["design_params"].to(trainer.device).float(),
                init_config=batch["init_config"].to(trainer.device).float(),
                target=batch["target_metrics"].to(trainer.device).float(),
            )

            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(batch["target_metrics"].detach().cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    metric_names = [
        "contact_norm",
        "disturbance_score",
        "angular_span_norm",
    ]

    log_dict = {}

    plot_dir = os.path.join(args.save_dir, "prediction_quality", split_name)
    os.makedirs(plot_dir, exist_ok=True)

    for i, name in enumerate(metric_names):
        y_true = true[:, i]
        y_pred = pred[:, i]

        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

        # Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            spearman = spearmanr(y_true, y_pred).correlation
            if np.isnan(spearman):
                spearman = 0.0
        except Exception:
            spearman = 0.0

        log_dict[f"{split_name}/{name}_mae"] = mae
        log_dict[f"{split_name}/{name}_rmse"] = rmse
        log_dict[f"{split_name}/{name}_r2"] = r2
        log_dict[f"{split_name}/{name}_spearman"] = spearman

        # predicted vs true scatter
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"{split_name}: Predicted vs True {name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        plt.plot([lo, hi], [lo, hi], "--")
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"{split_name}: Predicted vs True {name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        log_dict[f"{split_name}/{name}_scatter"] = wandb.Image(fig)

        plt.close(fig)

    # --------------------------------------------------
    # Top-K ranking validation for combined objective
    # --------------------------------------------------
    # Same objective as optimization: disturbance + contact + angular span
    pred_score = pred[:, 1] + 0.1 * pred[:, 0] + 0.5 * pred[:, 2]
    true_score = true[:, 1] + 0.1 * true[:, 0] + 0.5 * true[:, 2]

    for k in [5, 10, 20]:
        k = min(k, len(pred_score))

        pred_topk = np.argsort(pred_score)[-k:]
        true_topk = np.argsort(true_score)[-k:]

        overlap = len(set(pred_topk).intersection(set(true_topk))) / k
        true_perf_of_pred_topk = np.mean(true_score[pred_topk])
        true_perf_of_true_topk = np.mean(true_score[true_topk])

        log_dict[f"{split_name}/top{k}_overlap"] = overlap
        log_dict[f"{split_name}/top{k}_true_score_of_pred_topk"] = true_perf_of_pred_topk
        log_dict[f"{split_name}/top{k}_oracle_true_score"] = true_perf_of_true_topk

    wandb.log(log_dict, step=epoch)

    return log_dict

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

            batch_mae = torch.abs(pred - target)   # shape [B, 3]

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
        f'disturbance_score: {avg_mae_per_dim[1].item():.4f}, '
        f'angular_span: {avg_mae_per_dim[2].item():.4f}'
    )

    return avg_loss, avg_mae, avg_mae_per_dim

def train(args):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"squirrel_dynamics_{timestamp}"

    wandb.init(
        project="squirrel-gripper-dynamics",
        config=args,
        dir=args.save_dir,
        name=run_name,
    )
    from torch.utils.data import Subset

    train_dataset = DynamicsDataset(dataset_dir=args.data_dir)
    val_dataset = DynamicsDataset(dataset_dir=args.test_data_dir)

    # full_dataset = DynamicsDataset(dataset_dir=args.data_dir)
    # np.random.seed(0)
    # debug_indices = np.random.choice(len(full_dataset), 32, replace=False)
    # train_dataset = Subset(full_dataset, debug_indices)
    # val_dataset = Subset(full_dataset, debug_indices)

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
                    'val/mae_angular_span': avg_mae_per_dim[2].item(),
                })

                evaluate_prediction_quality(
                    trainer=trainer,
                    dataloader=val_loader,
                    args=args,
                    epoch=epoch,
                    split_name="val",
                )
                
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
