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
    total_mae = 0
    total_val_loss = 0
    tolerance = 0.1 
    correct_predictions = 0
    total_samples = 0
    trainer.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            score = batch['disturbance_params'].to(trainer.device).float()
            ctrlpts = batch['ctrlpts'].to(trainer.device).float()
            tension = batch['input_tension'].to(trainer.device).float()

            finger_dict = batch['finger_params']
            nodes = finger_dict['nodes'].to(trainer.device).float()
            base_length = finger_dict['base_length'].to(trainer.device).float()
            base_radius = finger_dict['base_radius'].to(trainer.device).float()
            input_ori = finger_dict['input_ori'].to(trainer.device).float()
            youngs_modulus = finger_dict['youngs_modulus'].to(trainer.device).float()
            finger_mass = finger_dict['finger_mass'].to(trainer.device).float() 
            body_mass = finger_dict['body_mass'].to(trainer.device).float()
            joint_softness = finger_dict['joint_softness'].to(trainer.device).float()

            cylinder_dict = batch['cylinder_params']
            cyl_position = cylinder_dict['cyl_position'].to(trainer.device).float()
            cyl_directors = cylinder_dict['cyl_directors'].to(trainer.device).float()
            cyl_radius = cylinder_dict['cyl_radius'].to(trainer.device).float()
            cyl_length = cylinder_dict['cyl_length'].to(trainer.device).float()

            contact_dict = batch['contact_params']
            nu_contact = contact_dict['nu_contact'].to(trainer.device).float()
            mu_contact = contact_dict['mu_contact'].to(trainer.device).float()
            
            pred, loss = trainer.inference(ctrlpts, tension, nodes, base_length, 
                                           base_radius, input_ori, youngs_modulus, 
                                           finger_mass, body_mass, joint_softness,
                                           cyl_position, cyl_directors, cyl_radius, cyl_length,
                                           nu_contact, mu_contact)
            
            batch_mae = torch.abs(pred - score)
            total_mae += batch_mae.mean().item()

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
                score = batch['disturbance_params'].to(trainer.device).float()
                
                # Input Unpacking (Mirroring your validate function)
                ctrlpts = batch['ctrlpts'].to(trainer.device).float()
                tension = batch['input_tension'].to(trainer.device).float()
                
                f_d = batch['finger_params']
                c_d = batch['cylinder_params']
                ct_d = batch['contact_params']

                loss, pred = trainer.step(
                    score, ctrlpts, tension, 
                    f_d['nodes'].to(trainer.device).float(),
                    f_d['base_length'].to(trainer.device).float(),
                    f_d['base_radius'].to(trainer.device).float(),
                    f_d['input_ori'].to(trainer.device).float(),
                    f_d['youngs_modulus'].to(trainer.device).float(),
                    f_d['finger_mass'].to(trainer.device).float(),
                    f_d['body_mass'].to(trainer.device).float(),
                    f_d['joint_softness'].to(trainer.device).float(),
                    c_d['cyl_position'].to(trainer.device).float(),
                    c_d['cyl_directors'].to(trainer.device).float(),
                    c_d['cyl_radius'].to(trainer.device).float(),
                    c_d['cyl_length'].to(trainer.device).float(),
                    ct_d['nu_contact'].to(trainer.device).float(),
                    ct_d['mu_contact'].to(trainer.device).float()
                )
                average_loss += loss

                if idx_batch % args.save_ckpt_step == 0:
                    trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pt'))

            trainer.lr_scheduler.step()
            
            # Logging
            if epoch % args.val_step == 0:
                val_loss, val_acc, avg_mae = validate(args, val_loader, trainer)
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