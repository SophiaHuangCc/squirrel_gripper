# Squirrel finger diffusion generator

This folder adapts the DGDM `generator` pattern to this project.

The generator learns a conditional design prior:

```text
task params + init config + desired metrics  --->  diffusion model  --->  16D From Links design
```

The 16D design is:

```text
[3 joint softness values,
 4 free-link lengths,
 3 finite joint lengths,
 base radius,
 base thickness,
 base length,
 tension,
 ankle wrap radius,
 ankle stiffness]
```

## Files

- `diffusion_utils.py`
  - DGDM-style 1D conditional UNet.
  - Predicts the diffusion noise residual for a noisy design vector.

- `dataloader.py`
  - Wraps `dynamics.dataloader.DynamicsDataset`.
  - Reuses the exact same data parsing as dynamics-model training.
  - Converts the model-normalized design vector back to physical units, then maps it to `[-1, 1]` for diffusion.
  - Projects generated free links so links plus joint lengths sum to `base_length`.

- `diffusion.py`
  - Main diffusion module.
  - Implements training loss, DDIM sampling, local EMA weights, and optional dynamics-model guidance.

- `train.py`
  - Trains the diffusion generator on your generated `.npz` dataset.
  - Saves `best.pt`, `last.pt`, periodic checkpoints, and `design_bounds.npz`.

- `sample.py`
  - Loads a trained diffusion checkpoint.
  - Generates candidate physical designs.
  - Optionally loads the trained dynamics model to guide/rerank samples.
  - Saves `generated_candidates.npz`.

- `train_diffusion.sh`
  - Example training command.

- `sample_diffusion.sh`
  - Example sampling command.

## Basic usage

From `sg_ws`:

```bash
python generator/train.py \
    --data_dir "runs/exp3" \
    --save_dir "generator/runs/diffusion_exp3"
```

Then sample:

```bash
python generator/sample.py \
    --diffusion_checkpoint_path "generator/runs/diffusion_exp3/best.pt" \
    --dynamics_checkpoint_path "PATH_TO_YOUR_DYNAMICS_CHECKPOINT.pt" \
    --save_dir "generator/runs/sample_exp3" \
    --num_samples 512
```

Start with `--guidance_scale 0.0`. After unguided samples look reasonable, try small guidance:

```bash
--guidance_scale 0.01
```

The generated `top_design_params` are physical design parameters, matching the style of the optimization output.
