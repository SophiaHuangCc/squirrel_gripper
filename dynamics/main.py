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

def validate(args, val_loader, trainer, threshold_std=[0.5, 0.5, 0.5]):
    print('--- Validation Step ---')
    average_val_loss = 0
    average_val_accuracy = 0
    
    trainer.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # Unpack Squirrel Gripper specific data
            score = batch['scores'].cuda()
            input_ori = batch['input_ori'].cuda()  # Approach Angle
            input_pos = batch['input_pos'].cuda()  # [Tension, BaseRad]
            ctrlpts = batch['ctrlpts'].cuda()      # Rod Geometry
            
            pred, loss = trainer.inference(None, ctrlpts, score, input_ori, input_pos, None)
            
            # Simple Binary Accuracy: Did we predict the success margin correctly?
            accuracy = torch.mean((pred > threshold_std[0]).float() == (score > threshold_std[0]).float())
            
            average_val_loss += loss
            average_val_accuracy += accuracy

    average_val_loss /= len(val_loader)
    average_val_accuracy /= len(val_loader)
    
    print(f'Val Loss: {average_val_loss:.4f} | Val Accuracy: {average_val_accuracy:.4f}')
    return average_val_loss, average_val_accuracy

def train(args):
    wandb.init(
        project='squirrel-gripper-dynamics',
        config=args,
        dir=args.save_dir,
        name=args.wandb_id,
    )

    # Initialize Dataset using the folders we set up in the previous steps
    train_dataset = DynamicsDataset(dataset_dir=args.data_dir)
    val_dataset = DynamicsDataset(dataset_dir=args.test_data_dir)
    
    # These thresholds define what we consider a "successful" grasp in training
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
                score = batch['scores'].cuda()
                input_ori = batch['input_ori'].cuda()
                input_pos = batch['input_pos'].cuda()
                ctrlpts = batch['ctrlpts'].cuda()

                loss, pred = trainer.step(ctrlpts, score, input_ori, input_pos, None)
                average_loss += loss

                if idx_batch % args.save_ckpt_step == 0:
                    trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pt'))

            trainer.lr_scheduler.step()
            
            # Logging
            if epoch % args.val_step == 0:
                val_loss, val_acc = validate(args, val_loader, trainer, threshold_std=threshold_std)
                wandb.log({'train/loss': average_loss/len(train_loader), 'val/loss': val_loss, 'val/acc': val_acc})
                
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