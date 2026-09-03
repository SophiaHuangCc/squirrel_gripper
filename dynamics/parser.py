import argparse

def parse():
    parser = argparse.ArgumentParser(description="Dynamics Model for Squirrel Gripper")
    
    # Execution Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "validate"])
    parser.add_argument("--device", type=str, default="cuda")
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="../runs/exp1", help="Path to simulation results")
    parser.add_argument("--test_data_dir", type=str, default="../runs/exp1", help="Path to validation data")
    parser.add_argument("--object_dir", type=str, default="../assets/objects", help="Path to object meshes/cylinders")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Where to save models")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    
    # Model Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer-only learning rate")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--task_dim", type=int, default=3)
    parser.add_argument("--design_dim", type=int, default=16)
    parser.add_argument("--init_dim", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Architecture Specifics
    parser.add_argument("--fingers_3d", action="store_true", help="Use 3D point clouds instead of 2D profiles")
    parser.add_argument("--object_max_num_vertices", type=int, default=512)
    parser.add_argument("--val_step", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--save_ckpt_step", type=int, default=500, help="Save every N batches")
    parser.add_argument("--wandb_id", type=str, default="squirrel_dynamics_01")
    parser.add_argument("--wandb_project", type=str, default="squirrel-gripper-dynamics")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--metric_loss_weights", default="1,1,1", metavar="C,D,A",
        help="Nonnegative regression-loss weights for contact, disturbance, and angular span.",
    )
    parser.add_argument(
        "--utility_weights", default="0.20,0.45,0.35", metavar="C,D,A",
        help="Utility weights used only by the optional pairwise ranking loss.",
    )
    parser.add_argument("--ranking_loss_weight", type=float, default=0.0)
    parser.add_argument("--ranking_margin", type=float, default=0.05)
    parser.add_argument(
        "--ranking_min_target_delta", type=float, default=0.05,
        help="Ignore pairs whose measured utility differs by less than this value.",
    )
    parser.add_argument(
        "--ranking_max_design_distance", type=float, default=0.0,
        help=(
            "Optional maximum L2 distance between normalized designs used by the "
            "pairwise ranking loss. Zero disables the distance filter."
        ),
    )
    parser.add_argument("--use_design_noise", action="store_true")
    parser.add_argument(
        "--num_train_timesteps", type=int, default=100,
        help="Diffusion training horizon used by the prior and noisy DGDM dynamics model.",
    )
    parser.add_argument(
        "--num_timesteps_per_batch", type=int, default=4,
        help="Independent noise levels sampled per clean design for DGDM dynamics training.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument(
        "--noise_timestep_sampling", choices=("uniform", "inference"), default="uniform",
        help=(
            "For noisy dynamics training, sample any training timestep or only the "
            "DDIM timesteps that will actually be used during inference."
        ),
    )
    parser.add_argument("--use_es", action="store_true", help="Use evolutionary strategy for training")
    parser.add_argument(
        "--cma_sigma",
        type=float,
        default=0.35,
        help="Initial CMA-ES step size in raw optimizer coordinates.",
    )
    parser.add_argument(
        "--cma_popsize",
        type=int,
        default=32,
        help="Number of 16D From Links candidate fingers evaluated per CMA-ES generation.",
    )
    parser.add_argument(
        "--cma_raw_bound",
        type=float,
        default=4.0,
        help="Symmetric bound on raw CMA-ES parameters before sigmoid mapping.",
    )
    parser.add_argument("--approach_deg", type=float, default=60.0)
    parser.add_argument("--init_approach_deg", type=float, default=60.0)
    parser.add_argument(
        "--approach_deg_min",
        type=float,
        default=45.0,
        help="Lower approach-angle bound; exp3 training samples start at 45 degrees.",
    )
    parser.add_argument(
        "--approach_deg_max",
        type=float,
        default=75.0,
        help="Upper approach-angle bound; exp3 training samples end at 75 degrees.",
    )
    parser.add_argument(
        "--optimization_objectives",
        type=str,
        default=(
            "disturbance,disturbance_contact,contact,angular_span,"
            "disturbance_span,disturbance_contact_span"
        ),
        help="Comma-separated objectives to run in profile optimization.",
    )
    parser.add_argument("--output_dim", type=int, default=3)

    return parser.parse_args()
