"""Common Sampler interface + metadata container."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Dict

import torch


@dataclass
class SamplerMetadata:
    """Provenance metadata for a sampler invocation.

    These fields drive:
      - On-disk output layout (via `test_origin`)
      - The value written to `sigma.txt` for the local-Gaussian scorer
        (via `effective_sigma`)
      - Logging + reproducibility (via `name` + `hyperparams`)
    """
    name: str                           # stable registry key, e.g. "ddad_native"
    effective_sigma: float              # σ for the local-Gaussian scorer to consume
    test_origin: str                    # on-disk dirname segment, e.g. "ddad_native"
    hyperparams: Dict[str, Any] = field(default_factory=dict)


class Sampler(ABC):
    """Common interface for all reconstruction samplers.

    Contract
    --------
    A concrete Sampler:
      * Takes a loaded UNet (already on the target device, in eval mode) and
        a plain dict of hyperparameters in its __init__.
      * Implements `sample(x, seed)` which returns a single reconstruction as
        a `(C, H, W)` float tensor in `[-1, 1]`. Given the same `x` + `seed`,
        output must be bitwise identical (determinism is required for
        reproducible results).
      * Exposes `metadata` as a `SamplerMetadata` instance describing the
        sampler for logging and downstream tooling.

    The ABC deliberately does NOT enforce anything about batching — samplers
    consume one image at a time. Drivers (e.g. tools/generate_recons.py) loop
    over samples + seeds and handle I/O.
    """

    @abstractmethod
    def sample(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        """Produce one reconstruction of `x`, seeded by `seed`.

        Args:
            x: input image, shape `(1, C, H, W)` or `(C, H, W)`, floats in `[-1, 1]`.
            seed: integer seed; the same seed + same `x` must yield the same output.

        Returns:
            Reconstruction as `(C, H, W)` float tensor in `[-1, 1]`, on any device.
        """
        ...

    def sample_batch(self, x: torch.Tensor, seeds: List[int]) -> torch.Tensor:
        """Reconstructions of one image `x` for several seeds, as `(len(seeds), C, H, W)`.

        Default: loop over `sample`. Samplers that can run the seeds as one
        batch override this; the contract is that row k equals
        `sample(x, seeds[k])` up to floating-point non-determinism of batched
        kernels, so batched and sequential runs are interchangeable.
        """
        return torch.stack([self.sample(x, s) for s in seeds], 0)

    @property
    @abstractmethod
    def metadata(self) -> SamplerMetadata:
        """Describe this sampler for output layout and logging."""
        ...


def _as_batched(x: torch.Tensor) -> torch.Tensor:
    """Ensure the input has a batch dim. `(C,H,W)` → `(1,C,H,W)`; passthrough otherwise."""
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"expected (C,H,W) or (B,C,H,W); got {tuple(x.shape)}")


def _unbatched(x: torch.Tensor) -> torch.Tensor:
    """Drop a leading batch dim of size 1. Passthrough otherwise."""
    if x.dim() == 4 and x.shape[0] == 1:
        return x.squeeze(0)
    return x
