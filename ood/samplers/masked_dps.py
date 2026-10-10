"""Masked (inpainting) DPS sampler: y = x + σε observed outside a mask.

Same reverse process as `additive_dps` (DDAD_DPS/samplers.py's
DPSPosteriorSampling), but the DPS residual ‖y − x̂₀‖ is restricted to the
observed region — pixels inside `inpaint_mask` are unobserved and get filled
in purely by the diffusion prior + boundary consistency.

Hyperparameters (passed as a dict to __init__):
    beta_start, beta_end     linear β schedule; default 0.0001, 0.02
    trajectory_steps         total DDPM steps; default 1000
    skip                     DDIM stride; default 25 (→ 40 reverse steps)
    eta                      DDIM stochasticity; default 1.0
    sigma                    additive measurement noise std (y = x + σε); default 0.1
    scale                    DPS guidance step size; default 0.5
    inpaint_mask             (H, W) array-like; nonzero = pixels to inpaint
                             (unobserved). None (default) = observe everything,
                             which degenerates to `additive_dps`.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

from .base import Sampler, SamplerMetadata, _as_batched, _unbatched


def _ensure_ddad_dps_on_path() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    p = str(root / "DDAD_DPS")
    if p not in sys.path:
        sys.path.insert(0, p)


def _make_dps_config(params: Dict[str, Any], device: str) -> SimpleNamespace:
    defaults = dict(
        beta_start=0.0001, beta_end=0.02,
        trajectory_steps=1000, skip=25, eta=1.0,
    )
    merged = {**defaults, **params, "device": device}
    return SimpleNamespace(model=SimpleNamespace(**merged))


class MaskedDPSSampler(Sampler):
    """DPS posterior sampling with a masked (inpainting) measurement model."""

    name = "masked_dps"

    def __init__(self, unet: torch.nn.Module, config: Dict[str, Any]):
        _ensure_ddad_dps_on_path()
        from samplers import DPSPosteriorSampling                # type: ignore

        self._device = config.get("device", next(unet.parameters()).device)
        self._device = str(self._device)
        self._sigma = float(config.get("sigma", 0.1))
        self._scale = float(config.get("scale", 0.5))

        mask = config.get("inpaint_mask", None)
        if mask is None:
            self._obs_mask = None
            self._inpaint_frac = 0.0
        else:
            m = np.asarray(mask)
            if m.ndim != 2:
                raise ValueError(f"inpaint_mask must be (H, W); got {m.shape}")
            inpaint = torch.from_numpy((m != 0).astype(np.float32))
            self._obs_mask = (1.0 - inpaint).view(1, 1, *m.shape).to(self._device)
            self._inpaint_frac = float(inpaint.mean().item())

        cfg = {k: v for k, v in config.items() if k != "inpaint_mask"}
        self._dps_cfg = _make_dps_config(cfg, self._device)
        self._sampler = DPSPosteriorSampling(unet, self._dps_cfg)

    def sample(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        x_clean = _as_batched(x).to(self._device)
        noise = torch.randn_like(x_clean)
        y = x_clean + self._sigma * noise
        xs = self._sampler(y, scale=self._scale, obs_mask=self._obs_mask)
        return _unbatched(xs[-1])                                # final x0

    @property
    def metadata(self) -> SamplerMetadata:
        m = self._dps_cfg.model
        return SamplerMetadata(
            name=self.name,
            effective_sigma=self._sigma,
            test_origin="masked_dps",
            hyperparams={
                "sigma": self._sigma,
                "scale": self._scale,
                "skip": m.skip,
                "eta": m.eta,
                "trajectory_steps": m.trajectory_steps,
                "beta_start": m.beta_start,
                "beta_end": m.beta_end,
                "inpaint_frac": self._inpaint_frac,
            },
        )
