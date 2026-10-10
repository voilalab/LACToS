"""Tests for ood/data.py — loaders for recons, labels, masks, GT."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ood.data import (
    load_reconstructions,
    load_label_image,
    load_gt_mask,
    load_superpixel_mask,
    parse_sample_id,
)


def _make_fake_recon_dir(root: Path, sample_name: str, test_origin: str,
                        patches, bottom_suffix: str, n_per_patch: int, size: int = 32):
    """Create a fake on-disk recon layout matching load_reconstructions expectations."""
    for p in patches:
        d = root / sample_name / f"{test_origin}_{p}_{bottom_suffix}" / "inpainting" / "recon"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_patch):
            arr = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}_{p}_00000.png")
        # Also add a label file so load_label_image can find one
        label_dir = root / sample_name / f"{test_origin}_{p}_{bottom_suffix}" / "inpainting" / "label"
        label_dir.mkdir(parents=True, exist_ok=True)
        lbl = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(lbl).save(label_dir / "0_00000.png")


def test_parse_sample_id_basic():
    assert parse_sample_id("samples_010") == "010"
    assert parse_sample_id("samples_40") == "40"
    assert parse_sample_id("samples_abc_123") == "abc_123"


def test_load_reconstructions_shape_and_dtype(tmp_path):
    _make_fake_recon_dir(
        tmp_path, "samples_000", "origin", patches=[0, 1, 2],
        bottom_suffix="4", n_per_patch=2, size=32,
    )
    recons = load_reconstructions(
        results_dir=str(tmp_path), sample_name="samples_000",
        test_origin="origin", num_patches=3, bottom_suffix="4",
    )
    assert recons.shape == (6, 32, 32, 3)  # 3 patches × 2 recons
    assert recons.dtype == np.uint8


def test_load_reconstructions_warns_on_missing(tmp_path, capsys):
    _make_fake_recon_dir(
        tmp_path, "samples_000", "origin", patches=[1, 2],
        bottom_suffix="4", n_per_patch=1, size=16,
    )
    # num_patches=3 expects patches 0,1,2 — patch 0 is missing
    recons = load_reconstructions(
        results_dir=str(tmp_path), sample_name="samples_000",
        test_origin="origin", num_patches=3, bottom_suffix="4",
    )
    captured = capsys.readouterr()
    assert "no images" in captured.out.lower() or "warning" in captured.out.lower()
    assert recons.shape[0] == 2  # patches 1 and 2 only


def test_load_label_image_returns_rgb(tmp_path):
    _make_fake_recon_dir(
        tmp_path, "samples_000", "origin", patches=[0],
        bottom_suffix="4", n_per_patch=1, size=32,
    )
    label = load_label_image(
        results_dir=str(tmp_path), sample_name="samples_000",
        test_origin="origin", bottom_suffix="4",
    )
    assert label.shape == (32, 32, 3)
    assert label.dtype == np.uint8


def test_load_gt_mask_downsamples(tmp_path):
    # Create a 64x64 GT mask with a 16x16 anomaly block, save as single-channel
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[16:32, 16:32] = 255
    path = tmp_path / "000_mask.png"
    Image.fromarray(gt, mode="L").save(path)

    mask = load_gt_mask(
        path_template=str(tmp_path / "{sample}_mask.png"),
        sample="000",
        downsample_factor=2,
    )
    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()).issubset({0, 1})


def test_load_superpixel_mask_8bit(tmp_path):
    """Plain uint8 mask with IDs 0..N (N<255) round-trips cleanly."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:32, :32] = 1
    mask[:32, 32:] = 2
    mask[32:, :32] = 3
    mask[32:, 32:] = 4
    Image.fromarray(mask, mode="L").save(tmp_path / "mask.png")
    loaded = load_superpixel_mask(str(tmp_path))
    assert loaded.shape == (64, 64)
    assert loaded.dtype == np.int32
    assert sorted(np.unique(loaded).tolist()) == [1, 2, 3, 4]


def test_load_superpixel_mask_16bit(tmp_path):
    """IDs > 255 must survive via the 16-bit fallback path."""
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[:32, :32] = 10
    mask[:32, 32:] = 500
    mask[32:, :32] = 1000
    mask[32:, 32:] = 2500
    Image.fromarray(mask, mode="I;16").save(tmp_path / "mask.png")
    loaded = load_superpixel_mask(str(tmp_path))
    assert loaded.shape == (64, 64)
    ids = set(np.unique(loaded).tolist())
    assert {10, 500, 1000, 2500} <= ids


def test_load_superpixel_mask_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_superpixel_mask(str(tmp_path))


def test_load_gt_mask_handles_rgba(tmp_path):
    """Faces GT masks are RGBA — our loader converts via .convert('L')."""
    gt_rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    gt_rgba[..., 3] = 255  # opaque
    gt_rgba[8:16, 8:16, :3] = 255  # white anomaly box in RGB channels
    path = tmp_path / "40_mask.png"
    Image.fromarray(gt_rgba, mode="RGBA").save(path)

    mask = load_gt_mask(
        path_template=str(tmp_path / "{sample}_mask.png"),
        sample="40",
        downsample_factor=1,
    )
    assert mask.shape == (32, 32)
    # The 8×8 white box in RGB becomes 1s; the rest 0s
    assert mask.sum() == 64
