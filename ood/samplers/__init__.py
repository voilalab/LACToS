"""Reconstruction samplers with a common interface.

All samplers satisfy the `Sampler` ABC in `base.py`. Every sampler is registered
in `SAMPLERS` below and instantiated via `build_sampler(name, unet, config)`.

Adding a new sampler:
  1. Create `ood/samplers/<name>.py` with a subclass of `Sampler`.
  2. Import it here and add to the `SAMPLERS` dict.
"""
from .base import Sampler, SamplerMetadata
from .additive_dps import AdditiveDPSSampler


SAMPLERS = {
    "additive_dps": AdditiveDPSSampler,
}


def build_sampler(name: str, unet, config: dict) -> Sampler:
    """Instantiate a sampler by name.

    Args:
        name: key in `SAMPLERS` (e.g. "ddad_native").
        unet: loaded PyTorch model (eval() mode, on target device).
        config: sampler-specific hyperparameters. Accepted keys depend on the
            concrete sampler; see each subclass.

    Returns:
        Configured `Sampler` instance.
    """
    if name not in SAMPLERS:
        raise ValueError(
            f"Unknown sampler {name!r}. Available: {sorted(SAMPLERS)}"
        )
    return SAMPLERS[name](unet, config)


__all__ = [
    "Sampler", "SamplerMetadata",
    "AdditiveDPSSampler",
    "SAMPLERS", "build_sampler",
]
