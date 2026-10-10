"""Pytest configuration and shared fixtures."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Markers ──
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_ckpt: test requires external DDAD checkpoint — skip with -m 'not requires_ckpt'",
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: test requires a CUDA-capable GPU",
    )
    config.addinivalue_line(
        "markers",
        "slow: test takes more than a few seconds",
    )


DDAD_CKPT_CABLE = str(ROOT / "assets" / "checkpoints" / "baselines" / "ffhq" / "ddad" / "unet.pth")  # the bundle ships the FFHQ DDAD UNet


def _ckpt_available() -> bool:
    return os.path.exists(DDAD_CKPT_CABLE)


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ── Auto-skip markers ──
def pytest_collection_modifyitems(config, items):
    skip_ckpt = pytest.mark.skip(reason="DDAD checkpoint not available")
    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    ckpt_ok = _ckpt_available()
    gpu_ok = _gpu_available()
    for item in items:
        if "requires_ckpt" in item.keywords and not ckpt_ok:
            item.add_marker(skip_ckpt)
        if "requires_gpu" in item.keywords and not gpu_ok:
            item.add_marker(skip_gpu)


# ── Shared fixtures ──
@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def tiny_scene(rng):
    """Small synthetic scene: 4 recons, 32x32 image, 3 channels, 4 SPs.

    Returns dict with:
      - labels_fine: (32, 32) int SP mask (4 regions)
      - parent_map: orig -> [child] (trivial identity)
      - images_recon_all: (4, 32, 32, 3) uint8
      - label_image: (32, 32, 3) uint8
    """
    H, W, C = 32, 32, 3
    N = 4
    # Four quadrant SPs
    labels = np.zeros((H, W), dtype=np.int32)
    labels[:H // 2, :W // 2] = 0
    labels[:H // 2, W // 2:] = 1
    labels[H // 2:, :W // 2] = 2
    labels[H // 2:, W // 2:] = 3

    # Label image: each SP has a distinct mean color + small noise
    label_image = np.zeros((H, W, C), dtype=np.float32)
    for sid, color in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]):
        label_image[labels == sid] = color
    label_image = np.clip(label_image + rng.normal(0, 5, label_image.shape), 0, 255).astype(np.uint8)

    # N reconstructions that roughly match the label but with per-seed variation
    images = np.stack([
        np.clip(label_image.astype(np.float32) + rng.normal(0, 10, label_image.shape), 0, 255).astype(np.uint8)
        for _ in range(N)
    ])

    return {
        "labels_fine": labels,
        "parent_map": {sid: [sid] for sid in range(4)},
        "images_recon_all": images,
        "label_image": label_image,
    }


@pytest.fixture(scope="session")
def ddad_ckpt_path():
    """Path to cable DDAD UNet checkpoint (if available)."""
    if not _ckpt_available():
        pytest.skip("DDAD checkpoint not available")
    return DDAD_CKPT_CABLE
