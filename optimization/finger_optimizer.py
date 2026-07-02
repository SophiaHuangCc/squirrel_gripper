import os
import sys
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from dynamics.sim_test import sim

import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from dynamics.profile_forward_2d import ProfileForward2DModel
from dynamics.parser import parse
from dynamics.metrics import metric2objective
from dynamics.utils import save_finger, finger_forward

os.environ['WANDB_CACHE_DIR'] = os.path.expanduser('~/.cache/wandb')
os.environ['WANDB_CONFIG_DIR'] = os.path.expanduser('~/.config/wandb')


def optimize_fingers(args, model, random_idx=80000):
    wandb.init(
        project='finger_optimize',
        job_type='optimize',
        config=args,
        dir=args.save_dir,
        mode='disabled' if getattr(args, 'disable_wandb', False) else 'online',
    )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # --------------------------------------------------
    # set up initialization
    # --------------------------------------------------
    if args.load_initialization:
        print('loading initialization')
        initial_data = np.load(os.path.join(args.data_dir, '%d.npz' % random_idx), allow_pickle=True)
        params_np = initial_data['raw_params']   # should be 13D if using approach optimization
        params = nn.Parameter(torch.Tensor(params_np).cuda(), requires_grad=True)
    else:
        print('using random initialization')
        if args.finger_sample == 'uniform':
            params = nn.Parameter(
                torch.Tensor(2. * torch.rand(args.optimizer_dim) - 1.0).cuda(),
                requires_grad=True,
            )
        elif args.finger_sample == 'gaussian':
            params = nn.Parameter(
                torch.Tensor(torch.randn(args.optimizer_dim)).cuda(),
                requires_grad=True,
            )
        else:
            raise ValueError('finger sample method not supported')

    initial_finger_dir = os.path.join(args.save_dir, 'initial')
    os.makedirs(initial_finger_dir, exist_ok=True)

    init_task_params, init_design_params = finger_forward(params, args)
    save_finger(
        init_design_params.detach().cpu().numpy(),
        initial_finger_dir,
        args,
        task_params=init_task_params.detach().cpu().numpy(),
    )

    wandb.log({'initial finger dir': initial_finger_dir})

    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam([params], lr=learning_rate)

    init_config = torch.Tensor(
        [[args.landing_height, args.landing_speed, args.initial_x_gap]]
    ).cuda()
    timesteps = torch.zeros((1,), dtype=torch.float32).cuda()

    # --------------------------------------------------
    # optimization loop
    # --------------------------------------------------
    last_best_epoch = 0
    best_params = None
    min_loss = torch.inf

    for i in tqdm(range(args.num_epochs)):
        # finger_forward now returns:
        # task_params   = [approach_deg, cyl_rad]
        # design_params = 12D physical design vector
        task_params, design_params = finger_forward(params, args)

        pred = model(
            task_params=task_params,
            design_params=design_params,
            init_config=init_config,
            timesteps=timesteps,
        )

        pred_contacts = pred[0, 0]
        pred_disturbance = pred[0, 1]
        pred_angular_span = pred[0, 2]
        pred_curl_speed = pred[0, 3]
        curl_quality_gate = torch.sigmoid(
            (pred_contacts - args.curl_contact_gate) / args.curl_gate_temperature
        )

        if args.optimization_loss == 'disturbance':
            loss = -pred_disturbance

        elif args.optimization_loss == 'disturbance_contact':
            loss = -(pred_disturbance + args.contact_weight * pred_contacts)

        elif args.optimization_loss == 'contact':
            loss = -pred_contacts

        elif args.optimization_loss == 'angular_span':
            loss = -args.angular_span_weight * pred_angular_span

        elif args.optimization_loss == 'disturbance_span':
            loss = -(
                args.disturbance_weight * pred_disturbance
                + args.angular_span_weight * pred_angular_span
            )

        elif args.optimization_loss == 'disturbance_contact_span':
            loss = -(
                args.disturbance_weight * pred_disturbance
                + args.contact_weight * pred_contacts
                + args.angular_span_weight * pred_angular_span
            )
        elif args.optimization_loss == 'curl_speed':
            loss = -pred_curl_speed

        elif args.optimization_loss == 'disturbance_contact_span_speed':
            quality = (
                args.disturbance_weight * pred_disturbance
                + args.contact_weight * pred_contacts
                + args.angular_span_weight * pred_angular_span
            )
            loss = -(
                quality
                + args.curl_speed_weight * pred_curl_speed * curl_quality_gate
            )

        else:
            raise ValueError('optimization loss not supported')

        if args.reg_weight > 0.0:
            loss = loss + args.reg_weight * torch.mean(params ** 2)

        wandb.log({
            'optimization loss': loss.item(),
            'pred_contacts': pred_contacts.detach().cpu().item(),
            'pred_disturbance': pred_disturbance.detach().cpu().item(),
            'pred_angular_span': pred_angular_span.detach().cpu().item(),
            'pred_curl_speed': pred_curl_speed.detach().cpu().item(),
            'opt_approach_deg': task_params[0, 0].detach().cpu().item(),
        })

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.learning_rate_decay == 0:
            learning_rate *= args.weight_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        if args.early_stopping:
            if loss.item() <= min_loss:
                min_loss = loss.item()
                best_params = params.detach().cpu().numpy()
                last_best_epoch = i
            elif i - last_best_epoch > args.patience:
                print('early stopping...')
                break
        else:
            best_params = params.detach().cpu().numpy()

        print(
            'Step:', i,
            'Loss:', loss.item(),
            'Pred contacts:', pred_contacts.detach().cpu().item(),
            'Pred disturbance:', pred_disturbance.detach().cpu().item(),
            'Pred angular span:', pred_angular_span.detach().cpu().item(),
            'Pred curl speed:', pred_curl_speed.detach().cpu().item(),
            'Optimized approach_deg:', task_params[0, 0].detach().cpu().item(),
        )

    print('Optimized raw params:', params)

    # --------------------------------------------------
    # save optimized finger params and test in simulation
    # --------------------------------------------------
    save_opt_finger_dir = os.path.join(args.save_dir, 'optimized')
    os.makedirs(save_opt_finger_dir, exist_ok=True)

    best_params_tensor = torch.Tensor(best_params).cuda()
    best_task_params, best_design_params = finger_forward(best_params_tensor, args)

    save_finger(
        best_design_params.detach().cpu().numpy(),
        save_opt_finger_dir,
        args,
        task_params=best_task_params.detach().cpu().numpy(),
    )

    wandb.log({'optimized finger dir': save_opt_finger_dir})

    metric, final_pose, video_path = sim(
        args.object_name,
        save_opt_finger_dir,
        os.path.join(save_opt_finger_dir, 'verification'),
        args.device_id,
        render=args.render_video,
        task_params=best_task_params.detach().cpu().numpy(),
    )

    wandb.log({'optimized_metric': metric})
    if args.render_video and video_path is not None:
        wandb.log({'optimized_video': wandb.Video(video_path)})

    metric_init, final_pose_init, video_path_init = sim(
        args.object_name,
        initial_finger_dir,
        os.path.join(initial_finger_dir, 'verification'),
        args.device_id,
        render=args.render_video,
        task_params=init_task_params.detach().cpu().numpy(),
    )

    wandb.log({'initial_metric': metric_init})
    if args.render_video and video_path_init is not None:
        wandb.log({'initial_video': wandb.Video(video_path_init)})

    # init_score = metric_init.get('disturbance_resistance_score', 0.0)
    # sim_score = metric.get('disturbance_resistance_score', 0.0)

    init_obj = metric2objective(metric_init, args.optimization_loss)
    sim_obj = metric2objective(metric, args.optimization_loss)

    init_score = init_obj.get("combined_score", list(init_obj.values())[0])
    sim_score = sim_obj.get("combined_score", list(sim_obj.values())[0])

    results = {
        'initial_metric': metric_init,
        'optimized_metric': metric,
        'initial_score': init_score,
        'optimized_score': sim_score,
        'initial_finger_dir': initial_finger_dir,
        'optimized_finger_dir': save_opt_finger_dir,
        'best_raw_params': best_params,
        'best_task_params': best_task_params.detach().cpu().numpy(),
        'best_design_params': best_design_params.detach().cpu().numpy(),
    }

    np.savez(os.path.join(args.save_dir, 'results.npz'), results)

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(
            {
                'initial_metric': metric_init,
                'optimized_metric': metric,
                'initial_score': float(init_score),
                'optimized_score': float(sim_score),
                'initial_finger_dir': initial_finger_dir,
                'optimized_finger_dir': save_opt_finger_dir,
            },
            f,
            indent=2,
        )

    wandb.finish()
    return init_score, sim_score


if __name__ == '__main__':
    args = parse()
    os.makedirs(args.save_dir, exist_ok=True)

    # Safe defaults if parser does not yet include these fields.
    if not hasattr(args, 'checkpoint_path'):
        args.checkpoint_path = getattr(args, 'ckpt_path', None)
    if not hasattr(args, 'model_type'):
        args.model_type = 'profile_forward'
    if not hasattr(args, 'design_dim'):
        args.design_dim = 13
    if not hasattr(args, 'task_dim'):
        args.task_dim = 2
    if not hasattr(args, 'init_dim'):
        args.init_dim = 3
    if not hasattr(args, 'output_dim'):
        args.output_dim = 4
    if not hasattr(args, 'hidden_dim'):
        args.hidden_dim = 256
    if not hasattr(args, 'design_dim'):
        args.design_dim = 13

    if not hasattr(args, 'finger_sample'):
        args.finger_sample = 'uniform'
    if not hasattr(args, 'load_initialization'):
        args.load_initialization = False
    if not hasattr(args, 'learning_rate'):
        args.learning_rate = 1e-2
    if not hasattr(args, 'num_epochs'):
        args.num_epochs = 300
    if not hasattr(args, 'learning_rate_decay'):
        args.learning_rate_decay = 100
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 0.5
    if not hasattr(args, 'early_stopping'):
        args.early_stopping = True
    if not hasattr(args, 'patience'):
        args.patience = 80
    if not hasattr(args, 'optimization_loss'):
        args.optimization_loss = 'disturbance_contact'
    if not hasattr(args, 'contact_weight'):
        args.contact_weight = 0.1
    if not hasattr(args, 'disturbance_weight'):
        args.disturbance_weight = 1.0
    if not hasattr(args, 'reg_weight'):
        args.reg_weight = 0.0
    if not hasattr(args, 'angular_span_weight'):
        args.angular_span_weight = 0.5
    if not hasattr(args, 'curl_speed_weight'):
        args.curl_speed_weight = 0.1
    if not hasattr(args, 'curl_contact_gate'):
        args.curl_contact_gate = 0.3
    if not hasattr(args, 'curl_gate_temperature'):
        args.curl_gate_temperature = 0.05

    if not hasattr(args, 'approach_deg'):
        args.approach_deg = 45.0
    if not hasattr(args, 'approach_deg_min'):
        args.approach_deg_min = 0.0
    if not hasattr(args, 'approach_deg_max'):
        args.approach_deg_max = 90.0
    if not hasattr(args, 'init_approach_deg'):
        args.init_approach_deg = args.approach_deg
    if not hasattr(args, 'cyl_rad'):
        args.cyl_rad = 0.03
    if not hasattr(args, 'landing_height'):
        args.landing_height = 0.04
    if not hasattr(args, 'landing_speed'):
        args.landing_speed = 0.0
    if not hasattr(args, 'initial_x_gap'):
        args.initial_x_gap = 0.06
    if not hasattr(args, 'object_name'):
        args.object_name = 'cylinder'
    if not hasattr(args, 'device_id'):
        args.device_id = 0
    if not hasattr(args, 'render_video'):
        args.render_video = False
    if not hasattr(args, 'disable_wandb'):
        args.disable_wandb = True

    if args.checkpoint_path is None:
        raise ValueError('Please provide --checkpoint_path or --ckpt_path.')

    # Load trained model
    if args.model_type == 'profile_forward':
        model = nn.DataParallel(
            ProfileForward2DModel(
                W=args.hidden_dim,
                task_ch=args.task_dim,
                design_ch=args.design_dim,
                init_ch=args.init_dim,
                output_ch=args.output_dim,
            ).cuda()
        )
        print('using squirrel profile forward model')
    else:
        raise ValueError('model type not supported')

    print('loading checkpoint from', args.checkpoint_path)
    model.load_state_dict(torch.load(args.checkpoint_path))

    initial_score, pred_score = optimize_fingers(args, model)
    average_increase = np.mean(np.asarray(pred_score) - np.asarray(initial_score))
    print('average increase:', average_increase)

    results = {
        'initial_score': np.asarray(initial_score),
        'pred_score': np.asarray(pred_score),
        'average_increase': average_increase,
    }
    np.savez(os.path.join(args.save_dir, 'score_results.npz'), results)
