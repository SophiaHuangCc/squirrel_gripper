import torch
import torch.nn as nn
import torch.optim as optim

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from dynamics.profile_forward_2d import ProfileForward2DModel


class Trainer:
    def __init__(self, args):
        self.args = args

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def create_model(self):
        """
        Inputs:
          - task_params   : [approach_angle, cylinder_radius]
          - design_params : [joint_stiffnesses, link_lengths, finger_radius,
                             finger_length, prebend_tension, ankle_radius,
                             ankle_stiffness]
          - init_config   : [drop_height, landing_speed, initial_x_gap]

        Output:
          - [num_contacts, force_closure, stability_margin]
        """

        self.task_dim = getattr(self.args, "task_dim", 2)
        self.design_dim = getattr(self.args, "design_dim", 12)
        self.init_dim = getattr(self.args, "init_dim", 3)
        self.output_dim = getattr(self.args, "output_dim", 2)
        self.hidden_dim = getattr(self.args, "hidden_dim", 256)

        # diffusion-style settings
        self.num_timesteps_per_batch = getattr(self.args, "num_timesteps_per_batch", 4)
        self.num_train_timesteps = getattr(self.args, "num_train_timesteps", 100)
        self.num_inference_steps = getattr(self.args, "num_inference_steps", 20)

        self.model = ProfileForward2DModel(
            W=self.hidden_dim,
            task_ch=self.task_dim,
            design_ch=self.design_dim,
            init_ch=self.init_dim,
            output_ch=self.output_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.criterion = nn.MSELoss()

        # DDIM scheduler only used to add noise to design params
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",  # fine to leave this, even though target is not epsilon
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

    def _prepare_tensors(self, task_params, design_params, init_config):
        task_tensor = task_params.view(task_params.shape[0], -1)
        design_tensor = design_params.view(design_params.shape[0], -1)
        init_tensor = init_config.view(init_config.shape[0], -1)
        return task_tensor, design_tensor, init_tensor

    def _make_noisy_design_batch(self, task_tensor, design_tensor, init_tensor, target):
        """
        Repeat the batch several times, sample timestep/noise, and add noise ONLY to design params.
        """
        B = design_tensor.shape[0]
        K = self.num_timesteps_per_batch

        task_all = task_tensor.repeat(K, 1)
        design_all = design_tensor.repeat(K, 1)
        init_all = init_tensor.repeat(K, 1)
        target_all = target.repeat(K, 1)

        noise = torch.randn_like(design_all, device=self.device)

        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(design_all.shape[0],),
            device=self.device,
        ).long()

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

        task_all, noisy_design_all, init_all, timestep_cond, target_all = self._make_noisy_design_batch(
            task_tensor, design_tensor, init_tensor, target
        )

        pred = self.model(
            task_params=task_all,
            design_params=noisy_design_all,
            init_config=init_all,
            timesteps=timestep_cond,
        )

        loss = self.criterion(pred, target_all)
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred

    def inference(self, task_params, design_params, init_config, target=None):
        self.model.eval()
        with torch.no_grad():
            prepared_inputs = self._prepare_tensors(task_params, design_params, init_config)

            # If you are repeating samples internally for diffusion/noise:
            task_tensor, design_tensor, init_tensor = prepared_inputs

            if getattr(self, "num_timesteps_per_batch", 1) > 1:
                B = task_tensor.shape[0]
                K = self.num_timesteps_per_batch

                task_all = task_tensor.repeat(K, 1)
                design_all = design_tensor.repeat(K, 1)
                init_all = init_tensor.repeat(K, 1)

                # however you currently add noise / timesteps:
                noise = torch.randn_like(design_all)
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (design_all.shape[0],),
                    device=self.device,
                ).long()

                noisy_design_all = self.noise_scheduler.add_noise(
                    original_samples=design_all,
                    noise=noise,
                    timesteps=timesteps,
                )
                timesteps = timesteps.float() / self.noise_scheduler.config.num_train_timesteps

                pred_all = self.model(task_all, noisy_design_all, init_all, timesteps)

                # reshape back to [K, B, output_dim], then average over K
                pred = pred_all.view(K, B, -1).mean(dim=0)

            else:
                pred = self.model(*prepared_inputs)

            if target is None:
                loss = torch.tensor(0.0, device=self.device)
            else:
                loss = self.criterion(pred, target)

            return pred, loss

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)