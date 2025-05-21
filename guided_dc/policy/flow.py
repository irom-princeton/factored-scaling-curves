"""
Conditional Flow matching

References:
https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb

"""

import logging
from collections import namedtuple

import numpy as np
import torch
from torch import nn

Sample = namedtuple("Sample", "trajectories chains")  # not saving intermediates
log = logging.getLogger(__name__)


def sample_from_transformed_beta(alpha, beta, s=0.999, size=1):
    """generate Gamma-distributed samples without using scipy, proposed by pi0 paper"""
    X = np.random.gamma(alpha, 1, size=size)
    Y = np.random.gamma(beta, 1, size=size)

    # compute Beta-distributed samples as Z = X / (X + Y)
    Z = X / (X + Y)

    # transform to τ domain
    tau_samples = s * (1 - Z)
    return tau_samples


class FlowModel(nn.Module):
    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="cuda:0",
        # Various clipping
        final_action_clip_value=None,
        # Flow parameters
        schedule="linear",
        num_inference_steps=10,
        gamma_alpha=1.5,
        gamma_beta=1,
        sig_min=0.001,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Flow parameters
        self.sig_min = sig_min
        self.num_inference_steps = num_inference_steps
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.gamma_max = 1 - sig_min
        assert schedule in ["linear", "gamma"], f"Invalid schedule: {schedule}"
        self.schedule = schedule

        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value

        # Set up models
        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded RL-trained policy from %s", network_path)
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(self, cond, deterministic=True):
        """Ignore deterministic flag"""
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=self.device)

        # forward euler integration
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(B, device=self.device)
        for _ in range(self.num_inference_steps):
            x += delta_t * self.network(x, t, cond)
            t += delta_t

        if self.final_action_clip_value is not None:
            x = torch.clamp(
                x, -self.final_action_clip_value, self.final_action_clip_value
            )
        return Sample(x, None)  # no chains

    # ---------- Supervised training ----------#

    def psi_t(self, x, x1, t):
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.sig_min) * t) * x + t * x1

    def loss(self, x1, cond):
        """Flow matching"""
        if self.schedule == "linear":  # uniform between 0 and 1
            eps = 1e-5
            t = (
                torch.rand(1, device=x1.device)
                + torch.arange(len(x1), device=x1.device) / len(x1)
            ) % (1 - eps)
        elif self.schedule == "gamma":  # pi0
            t = sample_from_transformed_beta(
                self.gamma_alpha, self.gamma_beta, self.gamma_max, size=len(x1)
            )
            t = torch.from_numpy(t).float().to(x1.device)  # (B,)
        # x ~ p_t(x0)
        x0 = torch.randn_like(x1)
        v_psi = self.network(self.psi_t(x0, x1, t), t, cond)
        d_psi = x1 - (1 - self.sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)


if "__main__" == __name__:
    import matplotlib.pyplot as plt
    import numpy as np

    # Parameters
    alpha = 1.5
    beta = 1
    s = 0.999
    size = 1000  # Number of samples

    # Generate samples
    tau_samples = sample_from_transformed_beta(alpha, beta, s, size)
    plt.hist(tau_samples, bins=30, density=True, alpha=0.7, color="blue")
    plt.xlabel("τ")
    plt.ylabel("Density")
    plt.title(f"Samples from Beta((s-τ)/s; {alpha}, {beta}) with s={s}")
    plt.grid(True)
    plt.savefig("beta_samples.png")
