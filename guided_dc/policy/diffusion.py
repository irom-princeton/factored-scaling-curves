"""
Gaussian diffusion with DDPM and optionally DDIM sampling.

References:
Diffuser: https://github.com/jannerm/diffuser
Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
Annotated DDIM/DDPM: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

"""

import logging
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from guided_dc.policy.common.sampling import (
    cosine_beta_schedule,
    extract,
    make_timesteps,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        horizon_steps,
        action_dim,
        model_path=None,
        device="cuda",
        # Various clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,  # DDIM only
        # DDPM parameters
        denoising_steps=100,
        predict_epsilon=True,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(device, int):
            device = f"cuda:{device}"
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Clip noise value at each denoising step
        self.denoised_clip_value = denoised_clip_value

        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Clip epsilon for numerical stability
        self.eps_clip_value = eps_clip_value

        # Set up models
        self.model = model.to(device)
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=True)
            else:
                self.load_state_dict(checkpoint["model"], strict=True)
            logging.info("Loaded checkpoint from %s", model_path)
        else:
            logging.info("Training from scratch")
        num_obs_encoder_params = sum(
            p.numel() for p in self.model.obs_encoder.parameters()
        )
        logging.info(
            f"Number of model parameters: {sum(p.numel() for p in self.parameters()) - num_obs_encoder_params}"
        )
        logging.info(f"Number of visual encoder parameters: {num_obs_encoder_params}")

        """
        DDPM parameters

        """
        """
        βₜ
        """
        self.betas = cosine_beta_schedule(denoising_steps).to(device)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ
        """
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]]
        )
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        r"""
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        r"""
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        """
        DDIM parameters

        In DDIM paper https://arxiv.org/pdf/2010.02502, alpha is alpha_cumprod in DDPM https://arxiv.org/pdf/2102.09672
        """
        if use_ddim:
            logging.info("Using DDIM")
            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == "uniform":  # use the HF "leading" style
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = (
                    torch.arange(0, ddim_steps, device=self.device) * step_ratio
                )
            else:
                raise ValueError("Unknown discretization method for DDIM.")
            self.ddim_alphas = (
                self.alphas_cumprod[self.ddim_t].clone().to(torch.float32)
            )
            self.ddim_alphas_sqrt = torch.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = torch.cat(
                [
                    torch.tensor([1.0]).to(torch.float32).to(self.device),
                    self.alphas_cumprod[self.ddim_t[:-1]],
                ]
            )
            self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5

            # Initialize fixed sigmas for inference - eta=0
            ddim_eta = 0
            self.ddim_sigmas = (
                ddim_eta
                * (
                    (1 - self.ddim_alphas_prev)
                    / (1 - self.ddim_alphas)
                    * (1 - self.ddim_alphas / self.ddim_alphas_prev)
                )
                ** 0.5
            )

            # Flip all
            self.ddim_t = torch.flip(self.ddim_t, [0])
            self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
            self.ddim_alphas_sqrt = torch.flip(self.ddim_alphas_sqrt, [0])
            self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch.flip(
                self.ddim_sqrt_one_minus_alphas, [0]
            )
            self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])

    # ---------- Sampling ----------#

    def p_mean_var(self, x, t, cond, index=None, model_override=None):
        if model_override is not None:
            noise = model_override(x, t, cond=cond)
        else:
            noise = self.model(x, t, cond=cond)

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                r"""
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε

            eta=0
            """
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = (1.0 - alpha_prev - sigma**2).sqrt() * noise
            mu = (alpha_prev**0.5) * x_recon + dir_xt
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    @torch.no_grad()
    def sample(self, cond, deterministic=True):
        """
        Forward pass for sampling actions. Used in evaluating pre-trained/fine-tuned policy. Not modifying diffusion clipping

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                std = torch.zeros_like(std)
            else:
                if t == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=1e-3)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return Sample(x, None)

    # ---------- Supervised training ----------#

    def forward(self, x, *args):
        batch_size = len(x)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, *args, t)

    def p_losses(
        self,
        x_start,
        cond: dict,
        t,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        device = x_start.device

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.model(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            return F.mse_loss(x_recon, noise, reduction="mean")
        else:
            return F.mse_loss(x_recon, x_start, reduction="mean")

    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        if noise is None:
            device = x_start.device
            noise = torch.randn_like(x_start, device=device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        model,
        horizon_steps,
        action_dim,
        denoising_steps,
        ddim_steps,
        predict_epsilon=True,
        model_path=None,
        device="cuda",
    ):
        super().__init__()
        if isinstance(device, int):
            device = f"cuda:{device}"
        self.device = device

        # Set up models
        self.model = model.to(device)
        if model_path is not None:
            checkpoint = torch.load(
                model_path, map_location=device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=True)
            else:
                self.load_state_dict(checkpoint["model"], strict=True)
            logging.info("Loaded checkpoint from %s", model_path)
        else:
            logging.info("Training from scratch")
        num_obs_encoder_params = sum(
            p.numel() for p in self.model.obs_encoder.parameters()
        )
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters()) - num_obs_encoder_params}"
        )
        logging.info(
            f"Number of visual encoder parameters: {num_obs_encoder_params}"
        )

        self.horizon_steps = horizon_steps
        self.action_dim = action_dim

        assert ddim_steps <= denoising_steps, "Can't eval with more steps!"
        self.denoising_steps = denoising_steps
        self.ddim_steps = ddim_steps
        self.predict_epsilon = predict_epsilon
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=denoising_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon" if predict_epsilon else "sample",
        )

    def forward(self, x, cond, mask_flat=None):
        B = len(x)
        t = torch.randint(0, self.denoising_steps, (B,), device=self.device).long()
        x = x.reshape((B, self.horizon_steps, self.action_dim))

        noise = torch.randn_like(x, device=self.device)

        # construct noise actions given real actions, noise, and diffusion schedule
        x_noisy = self.diffusion_schedule.add_noise(x, noise, t)
        _, x_recon = self.model(x_noisy, t, cond=cond)

        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction="none")
        else:
            loss = F.mse_loss(x_recon, x, reduction="none")
        if mask_flat:
            mask = mask_flat.reshape((B, self.horizon_steps, self.action_dim))
            loss = (loss * mask).sum(1)  # mask the loss to only consider "real" acs
        return loss.mean()

    @torch.no_grad()
    def sample(self, cond, deterministic=True, n_steps=None):
        enc_cache = None
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        x = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)

        # set number of steps
        eval_steps = self.ddim_steps
        if n_steps is not None:
            assert n_steps <= self.denoising_steps, f"can't be > {self.ddim_steps}"
            eval_steps = n_steps

        enc_cache = self.model.forward_enc(cond)

        # begin diffusion process
        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alphas_cumprod = (
            self.diffusion_schedule.alphas_cumprod.to(self.device)
        )
        for timestep in self.diffusion_schedule.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(self.device)
            noise_pred = self.model.forward_dec(x, batched_timestep, enc_cache)

            # take diffusion step
            x = self.diffusion_schedule.step(
                model_output=noise_pred, timestep=timestep, sample=x
            ).prev_sample

        # return final action post diffusion
        return Sample(x, None)
