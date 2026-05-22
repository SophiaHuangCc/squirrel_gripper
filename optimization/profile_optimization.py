import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
import numpy as np
import torch
import torch.nn as nn
import wandb

from dynamics.parser import parse
from dynamics.profile_forward_2d import ProfileForward2DModel
from dynamics.sim_test_mj import sim_test_batch
from dynamics.metrics import metric2objective
from profile_optimizer import ProfileOptimizer, ProfileOptimizerES


torch.multiprocessing.set_sharing_strategy("file_system")
# Can be commented out if already set up in dynamics model
os.environ['WANDB_CACHE_DIR'] = os.path.expanduser('~/.cache/wandb')
os.environ['WANDB_CONFIG_DIR'] = os.path.expanduser('~/.config/wandb')


def get_best_ids_all_metrics(objectives, opt_obj="disturbance"):
    if opt_obj == "disturbance":
        best_ids = {
            "disturbance_resistance_score": np.argmax(
                [objective["disturbance_resistance_score"] for objective in objectives]
            ),
        }
    elif opt_obj == "disturbance_contact":
        best_ids = {
            "disturbance_resistance_score": np.argmax(
                [objective["disturbance_resistance_score"] for objective in objectives]
            ),
            "num_contacts": np.argmax(
                [objective["num_contacts"] for objective in objectives]
            ),
            "combined_score": np.argmax(
                [objective["combined_score"] for objective in objectives]
            ),
        }
    elif opt_obj == "contact":
        best_ids = {
            "num_contacts": np.argmax(
                [objective["num_contacts"] for objective in objectives]
            ),
        }
    elif opt_obj == "angular_span":
        best_ids = {
            "angular_span": np.argmax(
                [objective["angular_span"] for objective in objectives]
            ),
        }

    elif opt_obj == "disturbance_span":
        best_ids = {
            "disturbance_resistance_score": np.argmax(
                [objective["disturbance_resistance_score"] for objective in objectives]
            ),
            "angular_span": np.argmax(
                [objective["angular_span"] for objective in objectives]
            ),
            "combined_score": np.argmax(
                [objective["combined_score"] for objective in objectives]
            ),
        }

    elif opt_obj == "disturbance_contact_span":
        best_ids = {
            "disturbance_resistance_score": np.argmax(
                [objective["disturbance_resistance_score"] for objective in objectives]
            ),
            "num_contacts": np.argmax(
                [objective["num_contacts"] for objective in objectives]
            ),
            "angular_span": np.argmax(
                [objective["angular_span"] for objective in objectives]
            ),
            "combined_score": np.argmax(
                [objective["combined_score"] for objective in objectives]
            ),
        }
    else:
        raise ValueError("opt obj not supported")

    return best_ids


import re
import datetime

def make_exp_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)

    exp_nums = []
    for name in os.listdir(base_dir):
        m = re.match(r"exp(\d+)$", name)
        if m:
            exp_nums.append(int(m.group(1)))

    next_id = max(exp_nums, default=0) + 1
    run_dir = os.path.join(base_dir, f"exp{next_id}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir
    

def optimization(args):
    input_spline_dim = 13   # 12 design params + 1 optimizable approach_deg
    num_spline_points = 1

    profile_model = ProfileForward2DModel(
        W=args.hidden_dim,
        task_ch=args.task_dim,
        design_ch=args.design_dim,
        init_ch=args.init_dim,
        output_ch=args.output_dim,
    ).cuda()

    print("loading profile network checkpoint from", args.checkpoint_path)
    profile_model.load_state_dict(torch.load(args.checkpoint_path))

    for param in profile_model.parameters():
        param.requires_grad = False
    profile_model.eval()

    for opt_obj in ["disturbance", "disturbance_contact", "contact", 
                    "angular_span", "disturbance_span", "disturbance_contact_span"]:
        if args.init_only:
            raise NotImplementedError(
                "init_only needs physical design params, not raw [-1, 1] params."
            )

        if args.use_es:
            optimizer_cls = ProfileOptimizerES
        else:
            optimizer_cls = ProfileOptimizer

        optimizer = optimizer_cls(
            profile_model=profile_model,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            opt_obj=opt_obj,
            input_dim=input_spline_dim,
            num_points=num_spline_points,
            grid_size=args.grid_size,
            object_vertices=None,
            seed=args.seed,
            device=torch.device("cuda:0"),

            approach_deg=args.approach_deg,
            cyl_radius=args.cyl_rad,
            landing_height=args.landing_height,
            landing_speed=args.landing_speed,
            initial_x_gap=args.initial_x_gap,

            contact_weight=args.contact_weight,
            disturbance_weight=args.disturbance_weight,
            reg_weight=args.reg_weight,
            angular_span_weight=args.angular_span_weight,

            joint_soft_min=args.joint_soft_min,
            joint_soft_max=args.joint_soft_max,
            link_min=args.link_min,
            link_max=args.link_max,
            base_radius_min=args.base_radius_min,
            base_radius_max=args.base_radius_max,
            base_length_min=args.base_length_min,
            base_length_max=args.base_length_max,
            tension_min=args.tension_min,
            tension_max=args.tension_max,
            ankle_wrap_min=args.ankle_wrap_min,
            ankle_wrap_max=args.ankle_wrap_max,
            ankle_stiff_min=args.ankle_stiff_min,
            ankle_stiff_max=args.ankle_stiff_max,

            init_joint_softness=args.init_joint_softness,
            init_link_lengths=args.init_link_lengths,
            init_base_radius=args.init_base_radius,
            init_base_length=args.init_base_length,
            init_tension=args.init_tension,
            init_ankle_wrap_radius=args.init_ankle_wrap_radius,
            init_ankle_stiffness=args.init_ankle_stiffness,
            init_approach_deg=args.init_approach_deg,

            approach_deg_min=args.approach_deg_min,
            approach_deg_max=args.approach_deg_max,
        )

        # design_result, task_result = optimizer.solve()
        design_result, task_result, pred_metrics = optimizer.solve()

        opt_save_dir = os.path.join(args.save_dir, f"{opt_obj}_surrogate_only")
        os.makedirs(opt_save_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(opt_save_dir, "optimized_candidates.npz"),
            design_params=design_result,
            task_params=task_result,
            pred_metrics=pred_metrics,
        )

        print(f"[SAVED] surrogate optimized candidates: {opt_save_dir}")

        if args.skip_sim:
            continue

        metrics, save_design_dirs = sim_test_batch(
            design_result,
            os.path.join(args.save_dir, f"{opt_obj}_cylinder"),
            render=args.render_video,
            num_cpus=args.num_cpus,
            task_params=task_result,
        )

        objectives = [metric2objective(metric, opt_obj) for metric in metrics]
        best_ids_all_metrics = get_best_ids_all_metrics(objectives, opt_obj)

        best_objectives = {
            k: objectives[best_ids_all_metrics[k]]
            for k in best_ids_all_metrics.keys()
        }

        best_design_dirs = {
            k: save_design_dirs[best_ids_all_metrics[k]]
            for k in best_ids_all_metrics.keys()
        }

        average_objectives = {
            k: np.mean([objective[k] for objective in objectives])
            for k in objectives[0].keys()
        }

        table = wandb.Table(
            columns=["metric_name", "best_objective", "best_finger_dir"],
            data=[
                ["average", average_objectives, "average"],
            ] + [
                [k, best_objectives[k], best_design_dirs[k]]
                for k in best_objectives.keys()
            ],
        )

        wandb.log({opt_obj: table})


if __name__ == "__main__":
    args = parse()

    # Safe defaults if parser does not yet include these fields.
    if not hasattr(args, "checkpoint_path"):
        args.checkpoint_path = getattr(args, "ckpt_path", None)
    if not hasattr(args, "save_dir"):
        args.save_dir = "finger_profile_optimization_results"
    if not hasattr(args, "hidden_dim"):
        args.hidden_dim = 256
    if not hasattr(args, "task_dim"):
        args.task_dim = 2
    if not hasattr(args, "design_dim"):
        args.design_dim = 12
    if not hasattr(args, "init_dim"):
        args.init_dim = 3
    if not hasattr(args, "output_dim"):
        args.output_dim = 3

    if not hasattr(args, "batch_size"):
        args.batch_size = 16
    if not hasattr(args, "num_epochs"):
        args.num_epochs = 300
    if not hasattr(args, "learning_rate"):
        args.learning_rate = 1e-4
    if not hasattr(args, "grid_size"):
        args.grid_size = 1
    if not hasattr(args, "seed"):
        args.seed = 0
    if not hasattr(args, "num_cpus"):
        args.num_cpus = 1
    if not hasattr(args, "render_video"):
        args.render_video = False
    if not hasattr(args, "init_only"):
        args.init_only = False
    if not hasattr(args, "use_es"):
        args.use_es = False

    if not hasattr(args, "approach_deg"):
        args.approach_deg = 45.0
    if not hasattr(args, "cyl_rad"):
        args.cyl_rad = 0.03
    if not hasattr(args, "landing_height"):
        args.landing_height = 0.04
    if not hasattr(args, "landing_speed"):
        args.landing_speed = 0.0
    if not hasattr(args, "initial_x_gap"):
        args.initial_x_gap = 0.06

    if not hasattr(args, "contact_weight"):
        args.contact_weight = 0.1
    if not hasattr(args, "disturbance_weight"):
        args.disturbance_weight = 1.0
    if not hasattr(args, "reg_weight"):
        args.reg_weight = 0.0
    if not hasattr(args, "angular_span_weight"):
        args.angular_span_weight = 0.5

    if not hasattr(args, "joint_soft_min"):
        args.joint_soft_min = 0.0005
    if not hasattr(args, "joint_soft_max"):
        args.joint_soft_max = 0.005
    if not hasattr(args, "link_min"):
        args.link_min = 0.02
    if not hasattr(args, "link_max"):
        args.link_max = 0.10
    if not hasattr(args, "base_radius_min"):
        args.base_radius_min = 0.01025
    if not hasattr(args, "base_radius_max"):
        args.base_radius_max = 0.013
    if not hasattr(args, "base_length_min"):
        args.base_length_min = 0.15
    if not hasattr(args, "base_length_max"):
        args.base_length_max = 0.25
    if not hasattr(args, "tension_min"):
        args.tension_min = 1.0
    if not hasattr(args, "tension_max"):
        args.tension_max = 6.0
    if not hasattr(args, "ankle_wrap_min"):
        args.ankle_wrap_min = 0.015
    if not hasattr(args, "ankle_wrap_max"):
        args.ankle_wrap_max = 0.025
    if not hasattr(args, "ankle_stiff_min"):
        args.ankle_stiff_min = 300.0
    if not hasattr(args, "ankle_stiff_max"):
        args.ankle_stiff_max = 700.0

    if not hasattr(args, "init_joint_softness"):
        args.init_joint_softness = [0.003, 0.003, 0.003]
    if not hasattr(args, "init_link_lengths"):
        args.init_link_lengths = [0.06, 0.056, 0.044, 0.04]
    if not hasattr(args, "init_base_radius"):
        args.init_base_radius = 0.01025
    if not hasattr(args, "init_base_length"):
        args.init_base_length = 0.20
    if not hasattr(args, "init_tension"):
        args.init_tension = 3.0
    if not hasattr(args, "init_ankle_wrap_radius"):
        args.init_ankle_wrap_radius = 0.02
    if not hasattr(args, "init_ankle_stiffness"):
        args.init_ankle_stiffness = 500.0
    if not hasattr(args, "init_approach_deg"):
        args.init_approach_deg = args.approach_deg
    if not hasattr(args, "approach_deg_min"):
        args.approach_deg_min = 0.0
    if not hasattr(args, "approach_deg_max"):
        args.approach_deg_max = 90.0

    if not hasattr(args, "skip_sim"):
        args.skip_sim = True

    if args.checkpoint_path is None:
        raise ValueError("Please provide --checkpoint_path or --ckpt_path")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Put each optimization run into runs/exp1, runs/exp2, ...
    base_runs_dir = os.path.join("optimization", "runs")
    args.save_dir = make_exp_dir(base_runs_dir)

    run_name = f"finger_opt_{os.path.basename(args.save_dir)}_{timestamp}"

    print("[OPT RUN DIR]", os.path.abspath(args.save_dir))
    print("[WANDB RUN NAME]", run_name)

    wandb.init(
        project="finger_profile_optimization",
        name=run_name,
        config=vars(args),
        dir=args.save_dir,
    )

    optimization(args)
    wandb.finish()
