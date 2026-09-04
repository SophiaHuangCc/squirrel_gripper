import torch
import torch.nn as nn
import torch.optim as optim

from dynamics.profile_forward_2d import ProfileForward2DModel
from generator.dataloader import DesignBounds, model_norm_to_physical, physical_to_diffusion


class Trainer:
    def __init__(self, args):
        self.args = args

        requested = getattr(args, "device", "cuda")
        if requested == "cpu":
            self.device = torch.device("cpu")
        elif requested == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif requested == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def create_model(self):
        """
        Inputs:
          - task_params   : [approach_angle, landing_approach_angle, cylinder_radius]
          - design_params : [joint_stiffnesses, link_lengths, joint_lengths, finger_radius,
                             finger_length, prebend_tension, ankle_radius,
                             ankle_stiffness]
          - init_config   : [drop_height, landing_speed, initial_x_gap]

        Output:
          - [num_contacts, disturbance_resistance_score, angular_span]
        """

        self.task_dim = getattr(self.args, "task_dim", 3)
        self.design_dim = getattr(self.args, "design_dim", 16)
        self.init_dim = getattr(self.args, "init_dim", 3)
        self.output_dim = getattr(self.args, "output_dim", 3)
        self.hidden_dim = getattr(self.args, "hidden_dim", 256)
        self.model_architecture = getattr(self.args, "model_architecture", "legacy")
        self.num_hidden_layers = int(getattr(self.args, "num_hidden_layers", 3))

        # diffusion-style settings
        self.num_timesteps_per_batch = getattr(self.args, "num_timesteps_per_batch", 4)
        self.num_train_timesteps = getattr(self.args, "num_train_timesteps", 100)
        self.num_inference_steps = getattr(self.args, "num_inference_steps", 20)
        self.use_design_noise = getattr(self.args, "use_design_noise", False)
        self.design_bounds = DesignBounds.defaults()

        self.model = ProfileForward2DModel(
            W=self.hidden_dim,
            task_ch=self.task_dim,
            design_ch=self.design_dim,
            init_ch=self.init_dim,
            output_ch=self.output_dim,
            architecture=self.model_architecture,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.metric_loss_weights = self._parse_weights(
            getattr(self.args, "metric_loss_weights", "1,1,1"),
            "--metric_loss_weights", require_sum_one=False,
        ).to(self.device)
        self.utility_weights = self._parse_weights(
            getattr(self.args, "utility_weights", "0.20,0.45,0.35"),
            "--utility_weights", require_sum_one=True,
        ).to(self.device)
        self.ranking_loss_weight = float(getattr(self.args, "ranking_loss_weight", 0.0))
        self.ranking_margin = float(getattr(self.args, "ranking_margin", 0.05))
        self.ranking_min_target_delta = float(
            getattr(self.args, "ranking_min_target_delta", 0.05)
        )
        self.ranking_max_design_distance = float(
            getattr(self.args, "ranking_max_design_distance", 0.0)
        )
        self.noise_timestep_sampling = getattr(
            self.args, "noise_timestep_sampling", "uniform"
        )
        noise_timestep_text = getattr(self.args, "noise_timesteps", "")
        self.noise_timesteps = tuple(
            int(value.strip()) for value in str(noise_timestep_text).split(",")
            if value.strip()
        )
        if any(value < 0 or value >= self.num_train_timesteps
               for value in self.noise_timesteps):
            raise ValueError(
                f"--noise_timesteps must be in [0, {self.num_train_timesteps - 1}]"
            )
        if (self.ranking_loss_weight < 0 or self.ranking_margin < 0
                or self.ranking_min_target_delta < 0
                or self.ranking_max_design_distance < 0):
            raise ValueError("Ranking-loss weight, margin, and target delta must be nonnegative")
        self.last_loss_parts = {}
        self.angular_target_normalization = getattr(
            self.args, "angular_target_normalization", "none"
        )
        self.angular_target_mean = 0.0
        self.angular_target_std = 1.0

        # Diffusers is needed only for the optional design-noise augmentation.
        # Keep ordinary surrogate training independent of the Hugging Face stack.
        self.noise_scheduler = None
        if self.use_design_noise:
            try:
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError(
                    "--use_design_noise requires a compatible diffusers and "
                    "huggingface_hub installation. Ordinary dynamics training "
                    "does not require this flag."
                ) from exc
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )
            self.noise_scheduler.set_timesteps(self.num_inference_steps)

    @staticmethod
    def _parse_weights(text, name, require_sum_one):
        values = [float(value.strip()) for value in str(text).split(",") if value.strip()]
        if len(values) != 3 or any(value < 0 for value in values) or sum(values) <= 0:
            raise ValueError(f"{name} must contain three nonnegative C,D,A values")
        if require_sum_one and abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"{name} must sum to one")
        return torch.tensor(values, dtype=torch.float32)

    def set_angular_target_statistics(self, mean, std):
        self.angular_target_mean = float(mean)
        self.angular_target_std = max(float(std), 1e-6)

    def transform_targets(self, values):
        if self.angular_target_normalization != "zscore":
            return values
        output = values.clone()
        output[..., 2] = (
            output[..., 2] - self.angular_target_mean
        ) / self.angular_target_std
        return output

    def inverse_predictions(self, values):
        if self.angular_target_normalization != "zscore":
            return values
        return torch.cat(
            (
                values[..., :2],
                values[..., 2:3] * self.angular_target_std + self.angular_target_mean,
            ),
            dim=-1,
        )

    def loss_components(
        self, prediction, target, task_params=None, init_config=None,
        design_params=None, timesteps=None,
    ):
        per_metric_mse = (prediction - target).square().mean(dim=0)
        regression = (
            per_metric_mse * self.metric_loss_weights
        ).sum() / self.metric_loss_weights.sum()
        ranking = prediction.sum() * 0.0
        pair_count = 0
        if self.ranking_loss_weight > 0 and len(prediction) > 1:
            predicted_utility = self.inverse_predictions(prediction) @ self.utility_weights
            target_utility = self.inverse_predictions(target) @ self.utility_weights
            # Compare all pairs, but only within the same task, initial condition,
            # and (for noisy training) timestep. Ranking across different physical
            # scenarios does not supervise the local design direction used by DGDM.
            predicted_delta = predicted_utility[:, None] - predicted_utility[None, :]
            target_delta = target_utility[:, None] - target_utility[None, :]
            valid = torch.triu(
                torch.ones_like(target_delta, dtype=torch.bool), diagonal=1
            )
            valid &= target_delta.abs() >= self.ranking_min_target_delta
            if task_params is not None:
                valid &= torch.isclose(
                    task_params[:, None, :], task_params[None, :, :], atol=1e-6, rtol=0.0
                ).all(dim=-1)
            if init_config is not None:
                valid &= torch.isclose(
                    init_config[:, None, :], init_config[None, :, :], atol=1e-6, rtol=0.0
                ).all(dim=-1)
            if timesteps is not None:
                valid &= torch.isclose(
                    timesteps[:, None], timesteps[None, :], atol=1e-7, rtol=0.0
                )
            if design_params is not None and self.ranking_max_design_distance > 0:
                distance = torch.linalg.vector_norm(
                    design_params[:, None, :] - design_params[None, :, :], dim=-1
                )
                valid &= distance <= self.ranking_max_design_distance
            if valid.any():
                signed_prediction = target_delta[valid].sign() * predicted_delta[valid]
                ranking = torch.relu(self.ranking_margin - signed_prediction).mean()
                pair_count = int(valid.sum().item())
        total = regression + self.ranking_loss_weight * ranking
        return total, {
            "total": total.detach(), "regression": regression.detach(),
            "ranking": ranking.detach(), "ranking_pairs": pair_count,
            "contact_mse": per_metric_mse[0].detach(),
            "disturbance_mse": per_metric_mse[1].detach(),
            "angular_span_mse": per_metric_mse[2].detach(),
        }

    def _prepare_tensors(self, task_params, design_params, init_config):
        task_tensor = task_params.view(task_params.shape[0], -1)
        design_tensor = design_params.view(design_params.shape[0], -1)
        init_tensor = init_config.view(init_config.shape[0], -1)
        return task_tensor, design_tensor, init_tensor

    def _make_noisy_design_batch(self, task_tensor, design_tensor, init_tensor, target):
        """
        Repeat the batch several times, sample timestep/noise, and add noise ONLY to design params.
        """
        # B = design_tensor.shape[0]
        # timesteps = torch.zeros(B, dtype=torch.float32, device=self.device)
        # return task_tensor, design_tensor, init_tensor, timesteps, target
        B = design_tensor.shape[0]

        if not self.use_design_noise:
            timesteps = torch.zeros(B, dtype=torch.long, device=self.device)
            return task_tensor, design_tensor, init_tensor, timesteps.float(), target

        K = self.num_timesteps_per_batch

        task_all = task_tensor.repeat(K, 1)
        # DGDM guidance acts on the diffusion state x_t, whose coordinates are
        # [-1, 1].  Corrupt the clean design in that same coordinate system.
        clean_unit = physical_to_diffusion(
            model_norm_to_physical(design_tensor), self.design_bounds
        ).clamp(-1.0, 1.0)
        design_all = clean_unit.repeat(K, 1)
        init_all = init_tensor.repeat(K, 1)
        target_all = target.repeat(K, 1)

        noise = torch.randn_like(design_all, device=self.device)

        if self.noise_timesteps:
            available = torch.tensor(
                self.noise_timesteps, dtype=torch.long, device=self.device
            )
            if K == len(available):
                # With K=3 and 0,3,6, every clean mini-batch is supervised at
                # every late guidance timestep rather than sampling duplicates.
                selected = available
            else:
                selected = available[
                    torch.randint(0, len(available), (K,), device=self.device)
                ]
        elif self.noise_timestep_sampling == "inference":
            available = self.noise_scheduler.timesteps.to(self.device)
            selected = available[
                torch.randint(0, len(available), (K,), device=self.device)
            ]
        else:
            selected = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(K,),
                device=self.device,
            )
        # One shared timestep per repeated batch permits valid within-context
        # ranking comparisons at an identical corruption level.
        timesteps = selected.repeat_interleave(B).long()

        noisy_design_all = self.noise_scheduler.add_noise(
            original_samples=design_all,
            noise=noise,
            timesteps=timesteps,
        )

        # normalize timestep to [0, 1], like the sample project
        timestep_cond = timesteps.float() / self.noise_scheduler.config.num_train_timesteps

        return task_all, noisy_design_all, init_all, timestep_cond, target_all

    def step(self, target, task_params, design_params, init_config):
        self.model.train()
        self.optimizer.zero_grad()

        task_tensor, design_tensor, init_tensor = self._prepare_tensors(
            task_params, design_params, init_config
        )

        task_tensor = task_tensor.to(self.device)
        design_tensor = design_tensor.to(self.device)
        init_tensor = init_tensor.to(self.device)
        target = target.to(self.device)

        target_model = self.transform_targets(target)
        task_all, noisy_design_all, init_all, timestep_cond, target_all = self._make_noisy_design_batch(
            task_tensor, design_tensor, init_tensor, target_model
        )

        pred = self.model(
            task_params=task_all,
            design_params=noisy_design_all,
            init_config=init_all,
            timesteps=timestep_cond,
        )

        loss, self.last_loss_parts = self.loss_components(
            pred, target_all, task_all, init_all, noisy_design_all, timestep_cond
        )
        loss.backward()

        # total_grad = 0.0
        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         total_grad += p.grad.abs().sum().item()

        # print("loss:", loss.item(), "total_grad:", total_grad)
        # print("pred[0]:", pred[0].detach().cpu().numpy())
        # print("target[0]:", target_all[0].detach().cpu().numpy())

        self.optimizer.step()

        return loss.item(), pred

    # def inference(self, task_params, design_params, init_config, target=None):
    #     self.model.eval()
    #     with torch.no_grad():
    #         prepared_inputs = self._prepare_tensors(task_params, design_params, init_config)
    #         task_tensor, design_tensor, init_tensor = prepared_inputs

    #         if getattr(self, "num_timesteps_per_batch", 1) > 1:
    #             B = task_tensor.shape[0]
    #             K = self.num_timesteps_per_batch

    #             task_all = task_tensor.repeat(K, 1)
    #             design_all = design_tensor.repeat(K, 1)
    #             init_all = init_tensor.repeat(K, 1)

    #             # however you currently add noise / timesteps:
    #             noise = torch.randn_like(design_all)
    #             timesteps = torch.randint(
    #                 0,
    #                 self.noise_scheduler.config.num_train_timesteps,
    #                 (design_all.shape[0],),
    #                 device=self.device,
    #             ).long()

    #             noisy_design_all = self.noise_scheduler.add_noise(
    #                 original_samples=design_all,
    #                 noise=noise,
    #                 timesteps=timesteps,
    #             )
    #             timesteps = timesteps.float() / self.noise_scheduler.config.num_train_timesteps

    #             pred_all = self.model(task_all, noisy_design_all, init_all, timesteps)

    #             # reshape back to [K, B, output_dim], then average over K
    #             pred = pred_all.view(K, B, -1).mean(dim=0)

    #         else:
    #             pred = self.model(*prepared_inputs)

    #         if target is None:
    #             loss = torch.tensor(0.0, device=self.device)
    #         else:
    #             loss = self.criterion(pred, target)

    #         return pred, loss
    def inference(self, task_params, design_params, init_config, target=None):
        self.model.eval()
        with torch.no_grad():
            task_tensor, design_tensor, init_tensor = self._prepare_tensors(
                task_params, design_params, init_config
            )

            task_tensor = task_tensor.to(self.device).float()
            design_tensor = design_tensor.to(self.device).float()
            init_tensor = init_tensor.to(self.device).float()

            if target is not None:
                target = target.to(self.device).float()
                target_model = self.transform_targets(target)

            if self.use_design_noise:
                dummy_target = target_model if target is not None else torch.zeros(
                    design_tensor.shape[0], self.output_dim, device=self.device
                )
                task_all, noisy_all, init_all, timestep_all, target_all = (
                    self._make_noisy_design_batch(
                        task_tensor, design_tensor, init_tensor, dummy_target
                    )
                )
                pred_all = self.model(task_all, noisy_all, init_all, timestep_all)
                pred_model = pred_all.reshape(
                    self.num_timesteps_per_batch, design_tensor.shape[0], self.output_dim
                ).mean(dim=0)
                pred = self.inverse_predictions(pred_model)
                loss = (
                    self.loss_components(pred_all, target_all)[0] if target is not None
                    else torch.tensor(0.0, device=self.device)
                )
            else:
                timesteps = torch.zeros(
                    design_tensor.shape[0], dtype=torch.float32, device=self.device
                )
                pred_model = self.model(task_tensor, design_tensor, init_tensor, timesteps)
                pred = self.inverse_predictions(pred_model)
                loss = (
                    self.loss_components(pred_model, target_model)[0] if target is not None
                    else torch.tensor(0.0, device=self.device)
                )

            return pred, loss

    def save_checkpoint(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "model_type": "aggregate_metric_dynamics",
            "noise_conditioned": bool(self.use_design_noise),
            "design_coordinates": "diffusion_unit" if self.use_design_noise else "model_norm",
            "num_train_timesteps": int(self.num_train_timesteps),
            "model_architecture": self.model_architecture,
            "hidden_dim": int(self.hidden_dim),
            "num_hidden_layers": int(self.num_hidden_layers),
            "angular_target_normalization": self.angular_target_normalization,
            "angular_target_mean": float(self.angular_target_mean),
            "angular_target_std": float(self.angular_target_std),
            "noise_beta_schedule": "squaredcos_cap_v2" if self.use_design_noise else None,
            "args": vars(self.args),
        }, path)
