# Squirrel DGDM

This is a separate, DGDM-style method. Unlike `generator/`, its diffusion prior is
unconditional and its dynamics network predicts a reusable interaction trajectory.
Tasks exist only as inference-time JSON profile targets evaluated over one or more
object/initial-condition scenarios.

```bash
python dgdm/train_prior.py --data_dir runs/exp3 --save_dir dgdm/runs/prior
python dgdm/train_dynamics.py --data_dir runs/exp3 --save_dir dgdm/runs/dynamics
python dgdm/generate.py --prior dgdm/runs/prior/last.pt \
  --dynamics dgdm/runs/dynamics/best.pt --task dgdm/task_example.json
```

Scenario columns are normalized `[approach_angle/90, cylinder_radius/0.05,
landing_height/0.10, landing_speed, initial_x_gap/0.10, friction, body_mass]`.
Profile channel names are stored in every dynamics checkpoint. Archives lacking a
signal use a zero training mask for that signal. This makes partial legacy data safe,
but profile objectives should only use channels well represented in the training set.

For the paper comparison, keep `generator/` as the conditional aggregate-metric
baseline. Compare unconditional prior samples, direct profile-gradient descent,
CMA-ES on the profile surrogate, the existing conditional generator, and this DGDM
sampler using identical scenario sets and simulator evaluation budgets.
