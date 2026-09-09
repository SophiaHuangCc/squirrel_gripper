#!/usr/bin/env bash
# Same training settings as V20's prior, but only six environment inputs are used.
set -Eeuo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python" -m generator.train \
  --data_dir "$PROJECT_DIR/TendonForces/runs/exp1/train" \
  --save_dir "$PROJECT_DIR/outputs/context_only_v20/diffusion_context_only_15" \
  --conditioning context_only --num_train_timesteps 15 --num_inference_steps 5 \
  --batch_size 512 --num_epochs 500 --num_workers 8 --learning_rate 0.0001 \
  --ema_power 0.75 --val_ratio 0.05 --seed 0 --device cuda \
  --save_every 25 --patience 30 --min_delta 0.00001 --wandb_mode disabled "$@"
