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
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Architecture Specifics
    parser.add_argument("--fingers_3d", action="store_true", help="Use 3D point clouds instead of 2D profiles")
    parser.add_argument("--object_max_num_vertices", type=int, default=512)
    parser.add_argument("--val_step", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--save_ckpt_step", type=int, default=500, help="Save every N batches")
    parser.add_argument("--wandb_id", type=str, default="squirrel_dynamics_01")
    parser.add_argument("--use_design_noise", action="store_true")
    parser.add_argument("--use_es", action="store_true", help="Use evolutionary strategy for training")
    parser.add_argument("--output_dim", type=int, default=4)
    parser.add_argument("--curl_contact_ratio", type=float, default=0.8)
    parser.add_argument("--curl_hold_time", type=float, default=0.2)
    parser.add_argument("--curl_min_contacts", type=int, default=3)
    parser.add_argument(
        "--curl_speed_weight", type=float, default=0.1,
        help="Weight of the gated curl-speed reward in the combined optimization objective.",
    )
    parser.add_argument(
        "--curl_contact_gate", type=float, default=0.3,
        help="Normalized predicted contact score at the midpoint of the speed quality gate.",
    )
    parser.add_argument(
        "--curl_gate_temperature", type=float, default=0.05,
        help="Softness of the predicted-contact gate applied to curl speed.",
    )

    return parser.parse_args()
