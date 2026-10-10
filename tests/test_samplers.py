"""Parametrized contract tests for every sampler in the registry.

Pattern-level tests (fast, no GPU, no checkpoint) verify that:
  - every registered sampler has the expected class attributes
  - every sampler's metadata has required fields
  - factory raises on unknown sampler names
  - I/O helpers produce the right layout

Runtime tests (slow, marked requires_ckpt + requires_gpu) verify:
  - a real sampler forward pass produces the right output shape
  - same (seed, input) ⇒ identical output (determinism)
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from ood.samplers import SAMPLERS, Sampler, SamplerMetadata, build_sampler
from ood.samplers.io import (
    effective_sigma_at_t_star, prepare_sample_dirs, recon_dir, tensor_to_uint8,
)


# ── fast contract tests ─────────────────────────────────────────────

def test_registry_is_nonempty_and_contains_expected_samplers():
    assert "ddad_native" in SAMPLERS
    assert "additive_dps" in SAMPLERS
    for cls in SAMPLERS.values():
        assert issubclass(cls, Sampler)


def test_build_sampler_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown sampler"):
        build_sampler("bogus_sampler", unet=None, config={})


def test_sampler_metadata_dataclass_fields():
    meta = SamplerMetadata(
        name="x", effective_sigma=0.1, test_origin="x_origin", hyperparams={"a": 1},
    )
    assert meta.name == "x"
    assert meta.effective_sigma == 0.1
    assert meta.test_origin == "x_origin"
    assert meta.hyperparams == {"a": 1}


def test_effective_sigma_at_t_star_matches_ddad_schedule():
    # t*=250 with DDAD's default schedule → ~0.69
    sigma = effective_sigma_at_t_star(250, trajectory_steps=1000,
                                      beta_start=0.0001, beta_end=0.02)
    assert 0.68 < sigma < 0.70
    # Monotonic: larger t* ⇒ larger σ
    assert effective_sigma_at_t_star(500) > sigma
    assert effective_sigma_at_t_star(100) < sigma


def test_tensor_to_uint8_round_trip():
    t = torch.stack([torch.full((2, 2), -1.0),
                     torch.full((2, 2),  0.0),
                     torch.full((2, 2),  1.0)], dim=0)
    arr = tensor_to_uint8(t)
    assert arr.shape == (2, 2, 3)
    assert arr.dtype == np.uint8
    # -1 → 0, 0 → 127/128, 1 → 255
    assert arr[0, 0, 0] == 0
    assert arr[0, 0, 2] == 255
    assert 126 <= arr[0, 0, 1] <= 128


def test_prepare_sample_dirs_creates_expected_layout(tmp_path):
    dirs = prepare_sample_dirs(tmp_path, "samples_000", "myrun", bottom_suffix="4")
    assert dirs["recon"].is_dir()
    assert dirs["label"].is_dir()
    assert dirs["input"].is_dir()
    # Path matches what ood/data.py loader expects
    expected = tmp_path / "samples_000" / "myrun_0_4" / "inpainting"
    assert recon_dir(tmp_path, "samples_000", "myrun", "4") == expected


# ── runtime tests (need GPU + checkpoint) ───────────────────────────

@pytest.fixture(scope="module")
def loaded_cable_unet():
    """One-time load of the cable UNet for sampler round-trip tests."""
    from ood.samplers.io import load_ddad_unet
    ckpt = str(Path(__file__).resolve().parent.parent / "assets" / "checkpoints" / "baselines" / "ffhq" / "ddad" / "unet.pth")
    import os
    if not os.path.exists(ckpt):
        pytest.skip("cable checkpoint not available")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    return load_ddad_unet(ckpt, device="cuda:0")


@pytest.mark.parametrize("name,params", [
    ("ddad_native", {"w": 2.0, "test_trajectoy_steps": 100, "skip": 25}),
    ("additive_dps", {"sigma": 0.1, "scale": 0.5, "skip": 25}),
])
@pytest.mark.requires_ckpt
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_sampler_produces_correct_shape_and_range(loaded_cable_unet, name, params):
    sampler = build_sampler(name, loaded_cable_unet, {**params, "device": "cuda:0"})
    x = torch.rand(1, 3, 256, 256, device="cuda:0") * 2 - 1
    out = sampler.sample(x, seed=7)
    assert out.shape == (3, 256, 256)
    assert out.dtype == torch.float32
    assert -2.0 < out.min().item() and out.max().item() < 2.0


@pytest.mark.parametrize("name,params", [
    ("ddad_native", {"w": 2.0, "test_trajectoy_steps": 100, "skip": 25}),
    ("additive_dps", {"sigma": 0.1, "scale": 0.5, "skip": 25}),
])
@pytest.mark.requires_ckpt
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_sampler_is_deterministic(loaded_cable_unet, name, params):
    """Same seed + same input → reconstruction stable to within cuDNN noise.

    DPS uses autograd through the UNet, which introduces sub-1e-2 non-determinism
    on CUDA even with seeds fixed (cuDNN non-deterministic algorithms). Downstream
    scoring quantizes to uint8, so anything <1/255 ≈ 4e-3 wouldn't change AUC.
    """
    sampler = build_sampler(name, loaded_cable_unet, {**params, "device": "cuda:0"})
    x = torch.rand(1, 3, 256, 256, device="cuda:0") * 2 - 1
    out1 = sampler.sample(x, seed=42)
    out2 = sampler.sample(x, seed=42)
    torch.testing.assert_close(out1, out2, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("name", list(SAMPLERS.keys()))
@pytest.mark.requires_ckpt
@pytest.mark.requires_gpu
def test_sampler_metadata_is_wellformed(loaded_cable_unet, name):
    sampler = build_sampler(name, loaded_cable_unet, {"device": "cuda:0"})
    meta = sampler.metadata
    assert isinstance(meta, SamplerMetadata)
    assert meta.name == name
    assert meta.test_origin
    assert 0.0 < meta.effective_sigma < 10.0
    assert isinstance(meta.hyperparams, dict) and meta.hyperparams
