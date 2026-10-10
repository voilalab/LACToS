"""
Diffusion Posterior Sampling (DPS) and DDAD-style reconstruction.

Mode 1 (DPS): Standard DPS from pure noise with gradient-based guidance.
Mode 2 (DDAD): Start from the diffusion timestep matching sigma, use DDAD conditioning.

Reference:
    DPS:  Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems", ICLR 2023.
    DDAD: Mousakhan et al., "Anomaly Detection with Conditioned Denoising Diffusion Models", WACV 2024.
"""

from typing import Any, List, Optional
import torch
import numpy as np
import os


def _build_alpha_bar(config, device):
    """Precompute cumulative alpha schedule (alpha_bar). Returns tensor of shape [T+1]."""
    betas = np.linspace(
        config.model.beta_start,
        config.model.beta_end,
        config.model.trajectory_steps,
        dtype=np.float64,
    )
    betas = torch.tensor(betas).type(torch.float).to(device)
    betas = torch.cat([torch.zeros(1).to(device), betas], dim=0)  # [T+1]
    alpha_bar = (1 - betas).cumprod(dim=0)  # [T+1], index 0 = alpha_bar_0 = 1
    return alpha_bar


def find_timestep_for_sigma(config, sigma, device):
    """
    Find the diffusion timestep t* such that sqrt(1 - alpha_bar_{t*}) ≈ sigma.

    Returns the closest valid timestep (clamped to [0, trajectory_steps-1]).
    """
    alpha_bar = _build_alpha_bar(config, device)
    # sqrt(1 - alpha_bar) for each timestep; index 0 is t=0, etc.
    noise_levels = (1 - alpha_bar[1:]).sqrt()  # [T], index i = timestep i
    # Find closest match
    t_star = (noise_levels - sigma).abs().argmin().item()
    return t_star


class DPSPosteriorSampling:
    """
    Mode 1: Standard DPS — start from pure noise, guide with grad ||y - x_0_hat||.
    """

    def __init__(self, unet, config) -> None:
        self.unet = unet
        self.config = config

    def __call__(
        self, y: torch.Tensor, scale: float = 1.0, obs_mask: torch.Tensor = None,
        generators: Optional[List[torch.Generator]] = None,
        noise_template: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            y: noisy measurement [B, C, H, W], where y = x_clean + sigma * noise.
            scale: DPS guidance step size.
            obs_mask: optional [1, 1, H, W] (or broadcastable) 0/1 mask; 1 marks
                observed pixels. The DPS residual is restricted to observed
                pixels (inpainting). None = all pixels observed.
            generators: optional list of B torch.Generators (on y's device), one
                per batch item. When given, every random draw for item b comes
                from generators[b] with a (1, C, H, W) shape, so a batch of B
                seeds reproduces B independent single-seed runs (the batch
                items never share an RNG stream). None = global RNG, as before.
            noise_template: (1, C, H, W) tensor whose strides the per-item draws
                copy (see _randn); defaults to y[0:1].
        Returns:
            List of intermediate samples [x_T, ..., x_0].
        """
        device = self.config.model.device
        alpha_bar = _build_alpha_bar(self.config, device)

        def _alpha(t):
            return alpha_bar.index_select(0, t + 1).view(-1, 1, 1, 1)

        n = y.size(0)
        eta = self.config.model.eta
        T = self.config.model.trajectory_steps  # full trajectory (1000)
        skip = self.config.model.skip

        if generators is not None and len(generators) != n:
            raise ValueError(f"need one generator per batch item: {len(generators)} vs B={n}")

        # With generators, each item's draw is `empty_like(template).normal_()`,
        # i.e. randn_like on a (1, C, H, W) tensor with the SAME STRIDES as the
        # single-seed path's y (the input image is channels-last after
        # image_to_tensor); randn fills memory order, so the strides decide
        # which value lands on which pixel. A contiguous randn of the same
        # shape gives a different image even with the same seed.
        template = noise_template if noise_template is not None else y[0:1]

        def _randn():
            if generators is None:
                return torch.randn_like(y)
            return torch.cat([torch.empty_like(template).normal_(generator=g)
                              for g in generators], 0)

        # Start from pure Gaussian noise (t = T-1)
        xt = _randn().to(device)

        seq = range(0, T, skip)
        seq_next = [-1] + list(seq[:-1])
        xs = [xt]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at = _alpha(t.long())
            at_next = _alpha(next_t.long())

            xt = xs[-1].to(device).requires_grad_(True)

            # Predict noise (with grad tracking for DPS)
            et = self.unet(xt, t)

            # Tweedie estimate
            x0_hat = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # DPS gradient: d/d(x_t) ||y - x_0_hat|| (restricted to observed pixels)
            difference = y.to(device) - x0_hat
            if obs_mask is not None:
                difference = difference * obs_mask.to(device)
            # Sum of PER-SAMPLE norms: its gradient w.r.t. x_t is the DPS
            # gradient of each item's own ||y_b - x0_hat_b||, so batch items
            # stay independent (a single norm over the whole batch would
            # couple them). For B = 1 this is exactly the previous code.
            norm = difference.flatten(1).norm(dim=1).sum()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            # Detach for the reverse step
            xt = xt.detach()
            et = et.detach()
            x0_hat = x0_hat.detach()

            # DDIM reverse step
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_hat + c1 * _randn() + c2 * et

            # DPS correction
            xt_next = xt_next - scale * norm_grad

            xs.append(xt_next)

        return xs


class DDADReconstruction:
    """
    Mode 2: DDAD-style — start from x_{t*} = sqrt(α̅_{t*}) * x_clean + sqrt(1-α̅_{t*}) * noise
    at the timestep matching sigma, then denoise with DDAD conditioning.
    """

    def __init__(self, unet, config) -> None:
        self.unet = unet
        self.config = config

    def __call__(
        self, x_clean: torch.Tensor, sigma: float, w: float
    ) -> List[torch.Tensor]:
        """
        Args:
            x_clean: clean image [B, C, H, W] (used to construct x_{t*} via forward process).
            sigma: noise level — determines the starting timestep t*.
            w: DDAD conditioning strength.
        Returns:
            List of intermediate samples [x_{t*}, ..., x_0].
        """
        device = self.config.model.device
        alpha_bar = _build_alpha_bar(self.config, device)

        def _alpha(t):
            return alpha_bar.index_select(0, t + 1).view(-1, 1, 1, 1)

        n = x_clean.size(0)
        eta = self.config.model.eta
        skip = self.config.model.skip

        # Find starting timestep from sigma
        t_star = find_timestep_for_sigma(self.config, sigma, device)
        at_star = _alpha(torch.tensor([t_star]).to(device).long())

        actual_sigma = (1 - at_star).sqrt().item()
        print(f"  t* = {t_star}  (requested sigma={sigma:.4f}, actual sqrt(1-α̅_t*)={actual_sigma:.4f})")

        # Construct x_{t*} via proper forward process
        xt = at_star.sqrt() * x_clean + (1 - at_star).sqrt() * torch.randn_like(x_clean).to(device)

        # Build timestep sequence up to t* (not the full test_trajectoy_steps)
        seq = range(0, t_star + 1, skip)
        seq_next = [-1] + list(seq[:-1])
        xs = [xt]

        # y0 for DDAD conditioning = x_clean (the reference image)
        y0 = x_clean.to(device)

        with torch.no_grad():
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(device)
                next_t = (torch.ones(n) * j).to(device)
                at = _alpha(t.long())
                at_next = _alpha(next_t.long())

                xt = xs[-1].to(device)

                # Predict noise
                et = self.unet(xt, t)

                # DDAD conditioning: stiffened prediction
                yt = at.sqrt() * y0 + (1 - at).sqrt() * et
                et_hat = et - (1 - at).sqrt() * w * (yt - xt)

                # Predict x0
                x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()

                # DDIM reverse step
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_clean) + c2 * et_hat

                xs.append(xt_next)

        return xs
