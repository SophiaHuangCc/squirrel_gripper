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

# def validate(args, val_loader, trainer, threshold_std=10):
#     print('--- Validation Step ---')
#     average_val_loss = 0
#     average_val_accuracy = 0
    
#     trainer.model.eval()
#     with torch.no_grad():
#         for batch in val_loader:
#             # Unpack Squirrel Gripper specific data
#             # score = batch['scores'].cuda()
#             # input_ori = batch['input_ori'].cuda()  # Approach Angle
#             # input_pos = batch['input_pos'].cuda()  # [Tension, BaseRad]
#             # ctrlpts = batch['ctrlpts'].cuda()      # Rod Geometry

#             score = batch['num_contacts'].to(trainer.device).float()
#             input_ori = batch['input_ori'].to(trainer.device).float()  # Approach Angle
#             input_pos = batch['input_pos'].to(trainer.device).float()  # [Tension, BaseRad]
#             ctrlpts = batch['ctrlpts'].to(trainer.device).float()      # Rod Geometry
#             obj_params = batch['obj_params'].to(trainer.device).float()
            
#             pred, loss = trainer.inference(None, ctrlpts, score, input_ori, input_pos, obj_params)

#             mae = torch.abs(pred - score).mean()
#             real_error = mae * 80
            
#             # Simple Binary Accuracy: Did we predict the success margin correctly?
#             accuracy = ((pred > threshold_std) == (score > threshold_std)).float().mean()
            
#             average_val_loss += loss
#             average_val_accuracy += accuracy

#     average_val_loss /= len(val_loader)
#     average_val_accuracy /= len(val_loader)
    
#     print(f'Val Loss: {average_val_loss:.4f} | Val Accuracy: {average_val_accuracy:.4f}')
#     return average_val_loss, average_val_accuracy

def validate(args, val_loader, trainer, threshold_std=10):
    print('--- Validation Step ---')
    total_mae = 0
    total_val_loss = 0
    
    # Define a tolerance for "New Accuracy" 
    # (e.g., predicting within 5 contacts of the real value)
    tolerance = 5.0 
    correct_predictions = 0
    total_samples = 0

    trainer.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            score = batch['num_contacts'].to(trainer.device).float()
            input_ori = batch['input_ori'].to(trainer.device).float()  # Approach Angle
            input_pos = batch['input_pos'].to(trainer.device).float()  # [Tension, BaseRad]
            ctrlpts = batch['ctrlpts'].to(trainer.device).float()      # Rod Geometry
            design_dict = batch['design_params']
            physics_dict = batch['physics_params']
            obj_dict = batch['obj_params']
            nodes = design_dict['nodes'].to(trainer.device).float()
            stiffness = physics_dict['stiffness'].to(trainer.device).float()
            cyl_rad = obj_dict['cyl_radius'].to(trainer.device).float()
            # design_params = batch['design_params'].to(trainer.device).float()
            # physics_params = batch['physics_params'].to(trainer.device).float()
            # obj_params = batch['obj_params'].to(trainer.device).float()
            
            pred, loss = trainer.inference(None, ctrlpts, score, input_ori, input_pos, nodes, stiffness, cyl_rad)
            
            # 1. Compute MAE (Average distance from the truth)
            batch_mae = torch.abs(pred - score)
            total_mae += batch_mae.mean().item()

            # 2. Compute "New Accuracy"
            # How many predictions fell within our tolerance?
            within_tolerance = (batch_mae < tolerance).float()
            correct_predictions += within_tolerance.sum().item()
            total_samples += score.size(0)

            total_val_loss += loss.item()

    avg_loss = total_val_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    accuracy = correct_predictions / total_samples
    
    print(f'Val Loss: {avg_loss:.4f} | MAE (Error in Contacts): {avg_mae:.2f} | Acc (within {tolerance} pts): {accuracy:.4f}')
    
    return avg_loss, accuracy, avg_mae

def train(args):
    wandb.init(
        project='squirrel-gripper-dynamics',
        config=args,
        dir=args.save_dir,
        name=args.wandb_id,
    )

    train_dataset = DynamicsDataset(dataset_dir=args.data_dir)
    val_dataset = DynamicsDataset(dataset_dir=args.test_data_dir)
    
    threshold_std = train_dataset.threshold 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    trainer = Trainer(args)
    trainer.create_model()

    if args.mode == 'validate':
        validate(args, val_loader, trainer, threshold_std=threshold_std)
        return

    if args.mode == 'train':
        best_val_loss = float('inf')
        last_best_epoch = 0
        
        for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
            average_loss = 0
            
            for idx_batch, batch in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
                # score = batch['scores'].cuda()
                # input_ori = batch['input_ori'].cuda()
                # input_pos = batch['input_pos'].cuda()
                # ctrlpts = batch['ctrlpts'].cuda()
                # obj_params = batch['obj_params'].cuda()
                # print("keys in batch:", batch.keys())

                score = batch['num_contacts'].to(trainer.device).float()
                input_ori = batch['input_ori'].to(trainer.device).float()
                input_pos = batch['input_pos'].to(trainer.device).float()
                ctrlpts = batch['ctrlpts'].to(trainer.device).float()
                design_dict = batch['design_params']
                physics_dict = batch['physics_params']
                obj_dict = batch['obj_params']
                nodes = design_dict['nodes'].to(trainer.device).float()
                stiffness = physics_dict['stiffness'].to(trainer.device).float()
                cyl_rad = obj_dict['cyl_radius'].to(trainer.device).float()
                # physics_params = batch['physics_params'].to(trainer.device).float()
                # obj_params = batch['obj_params'].to(trainer.device).float()
                # print(f"Nodes shape: {nodes.shape}")
                # print(f"Nodes flat shape: {nodes.view(nodes.shape[0], -1).shape}")

                loss, pred = trainer.step(ctrlpts, score, input_ori, input_pos, 
                                          nodes, stiffness, cyl_rad)
                average_loss += loss

                if idx_batch % args.save_ckpt_step == 0:
                    trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pt'))

            trainer.lr_scheduler.step()
            
            # Logging
            if epoch % args.val_step == 0:
                val_loss, val_acc, avg_mae = validate(args, val_loader, trainer, threshold_std=threshold_std)
                wandb.log({'train/loss': average_loss/len(train_loader), 
                           'val/loss': val_loss, 'val/acc': val_acc,
                           'val/avg_mae': avg_mae})
                
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