"""DDAD-native sampler: conditioned denoising starting from x_{t*}.

Thin wrapper around DDAD/reconstruction.py's Reconstruction class. See
DDAD/reconstruction.py for the algorithm (Algorithm 1 in Mousakhan et al.,
WACV 2024).

Hyperparameters (passed as a dict to __init__):
    beta_start, beta_end         linear β schedule; default 0.0001, 0.02
    trajectory_steps             DDPM training length; default 1000
    test_trajectoy_steps (t*)    starting timestep of reverse process; default 250
                                 (typo preserved to match DDAD's upstream config)
    skip                         DDIM stride; default 25 (→ 10 reverse steps)
    eta                          DDIM stochasticity; default 1.0 (DDPM-like)
    w                            conditioning strength toward y0; default 2.0
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

from .base import Sampler, SamplerMetadata, _as_batched, _unbatched
from .io import effective_sigma_at_t_star, _ensure_ddad_on_path


def _make_ddad_config(params: Dict[str, Any], device: str) -> SimpleNamespace:
    """Build a DDAD-style config object (`cfg.model.<field>`) from a flat dict."""
    defaults = dict(
        beta_start=0.0001, beta_end=0.02,
        trajectory_steps=1000, test_trajectoy_steps=250,
        skip=25, eta=1.0,
    )
    merged = {**defaults, **params, "device": device}
    return SimpleNamespace(model=SimpleNamespace(**merged))


class DDADNativeSampler(Sampler):
    """DDAD's conditioned denoising sampler (wraps DDAD.reconstruction.Reconstruction)."""

    name = "ddad_native"

    def __init__(self, unet: torch.nn.Module, config: Dict[str, Any]):
        _ensure_ddad_on_path()
        from reconstruction import Reconstruction              # type: ignore

        self._device = config.get("device", next(unet.parameters()).device)
        self._device = str(self._device)
        self._w = float(config.get("w", 2.0))
        self._t_star = int(config.get("test_trajectoy_steps", 250))

        self._ddad_cfg = _make_ddad_config(config, self._device)
        self._recon = Reconstruction(unet, self._ddad_cfg)

    def sample(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        x_in = _as_batched(x).to(self._device)
        xs = self._recon(x_in, x_in, self._w)
        return _unbatched(xs[-1])                              # final x0, drop batch dim

    @property
    def metadata(self) -> SamplerMetadata:
        m = self._ddad_cfg.model
        sigma = effective_sigma_at_t_star(
            self._t_star, m.trajectory_steps, m.beta_start, m.beta_end,
        )
        return SamplerMetadata(
            name=self.name,
            effective_sigma=sigma,
            test_origin="ddad_native",
            hyperparams={
                "w": self._w,
                "test_trajectoy_steps": self._t_star,
                "skip": m.skip,
                "eta": m.eta,
                "trajectory_steps": m.trajectory_steps,
                "beta_start": m.beta_start,
                "beta_end": m.beta_end,
            },
        )
