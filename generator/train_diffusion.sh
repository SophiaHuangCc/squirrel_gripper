python generator/train.py \
    --data_dir "runs/exp3" \
    --save_dir "generator/runs/diffusion_exp3" \
    --batch_size 512 \
    --num_epochs 500 \
    --num_train_timesteps 100 \
    --num_inference_steps 20 \
    --learning_rate 1e-4 \
    --num_workers 4

