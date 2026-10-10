"""Tests for ood/scoring.py (typical-set PMF) and ood/scoring_local_gaussian.py."""
import numpy as np
import pytest

from ood.scoring import compute_delta_map, quantize_u8_to_bins, quantize_pca_to_bins, joint_nd_pmf
from ood.scoring_local_gaussian import compute_delta_map_local_gaussian, SUPPORTED_METRICS


# ── Quantization ──
def test_quantize_u8_to_bins_basic():
    x = np.array([0, 128, 255], dtype=np.uint8)
    q = quantize_u8_to_bins(x, bins=16)
    assert q[0] == 0
    assert 0 <= q[1] < 16
    assert q[2] == 15


def test_quantize_pca_to_bins_clips():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    q = quantize_pca_to_bins(x, bins=8)
    assert q[0] == 0      # -2 clipped to -1 → bin 0
    assert q[-1] == 7     # 2 clipped to 1 → bin 7
    assert all(0 <= v < 8 for v in q)


def test_joint_nd_pmf_sums_to_one():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 8, size=(100, 3)).astype(np.uint8)
    pca = rng.integers(0, 4, size=(100, 3)).astype(np.uint8)
    p_rgb, p_pca = joint_nd_pmf(rgb, pca, bins_rgb=8, bins_pca=4, smooth_sigma=0.0)
    assert p_rgb.shape == (8, 8, 8)
    assert p_pca.shape == (4, 4, 4)
    assert p_rgb.sum() == pytest.approx(1.0, abs=1e-4)
    assert p_pca.sum() == pytest.approx(1.0, abs=1e-4)


# ── Typical-set PMF ──
def test_typical_set_returns_expected_shape(tiny_scene):
    # Fabricate PCA feats: same shape as recons but with k=2 PCA channels
    N, H, W, _ = tiny_scene["images_recon_all"].shape
    pca_feats = np.random.uniform(-1, 1, size=(N, H, W, 2)).astype(np.float32)
    label_pca = np.random.uniform(-1, 1, size=(H, W, 2)).astype(np.float32)

    delta_map, info, labels_used = compute_delta_map(
        labels_fine=tiny_scene["labels_fine"],
        parent_map=tiny_scene["parent_map"],
        images_recon_all=tiny_scene["images_recon_all"],
        pca_feats_recon=pca_feats,
        label_image=tiny_scene["label_image"],
        label_pca_map=label_pca,
        bins_rgb=8, bins_pca=4,
        smooth_sigma=0.1, min_pixels=2,
        use_label_as_target=True, eps=1e-12,
    )
    assert delta_map.shape == (H, W)
    assert delta_map.dtype == np.float32
    assert len(labels_used) > 0
    # Every scored SP should have an entry in info
    for sid in labels_used:
        assert sid in info
        assert "delta_sp" in info[sid]


def test_typical_set_bins_pca_1_disables_pca_contribution(tiny_scene):
    """With bins_pca=1 the PCA PMF collapses to [1.0] → no PCA contribution."""
    N, H, W, _ = tiny_scene["images_recon_all"].shape
    pca_feats = np.random.uniform(-1, 1, size=(N, H, W, 3)).astype(np.float32)
    label_pca = np.random.uniform(-1, 1, size=(H, W, 3)).astype(np.float32)

    delta_map, info, _ = compute_delta_map(
        labels_fine=tiny_scene["labels_fine"],
        parent_map=tiny_scene["parent_map"],
        images_recon_all=tiny_scene["images_recon_all"],
        pca_feats_recon=pca_feats,
        label_image=tiny_scene["label_image"],
        label_pca_map=label_pca,
        bins_rgb=8, bins_pca=1,  # degenerate
        smooth_sigma=0.1, min_pixels=2,
        use_label_as_target=True, eps=1e-12,
    )
    # With bins_pca=1, every PCA value maps to bin 0 → H_pca=0, nll_pca=0
    # So delta = |nll_rgb - H_rgb|. Shape should still be valid.
    assert delta_map.shape == (H, W)
    assert np.any(np.isfinite(delta_map))


# ── Local-Gaussian scorer ──
def test_local_gaussian_returns_expected_shape(tiny_scene):
    delta_map, info, labels_used = compute_delta_map_local_gaussian(
        labels_fine=tiny_scene["labels_fine"],
        parent_map=tiny_scene["parent_map"],
        images_recon_all=tiny_scene["images_recon_all"],
        label_image=tiny_scene["label_image"],
        sigma=0.1,
        n_realizations=100,
        min_pixels=2,
        metric="typicality_unsigned",
    )
    H, W = tiny_scene["labels_fine"].shape
    assert delta_map.shape == (H, W)
    assert len(labels_used) > 0


def test_local_gaussian_all_supported_metrics_work(tiny_scene):
    """Each metric variant should return a valid delta map."""
    for metric in SUPPORTED_METRICS:
        delta_map, _, labels_used = compute_delta_map_local_gaussian(
            labels_fine=tiny_scene["labels_fine"],
            parent_map=tiny_scene["parent_map"],
            images_recon_all=tiny_scene["images_recon_all"],
            label_image=tiny_scene["label_image"],
            sigma=0.1,
            n_realizations=100,
            min_pixels=2,
            metric=metric,
        )
        assert delta_map.shape == tiny_scene["labels_fine"].shape
        # At least the scored SPs should have finite values
        scored_mask = np.isin(tiny_scene["labels_fine"], labels_used)
        assert np.all(np.isfinite(delta_map[scored_mask]))


def test_local_gaussian_rejects_unknown_metric(tiny_scene):
    with pytest.raises(ValueError, match="metric"):
        compute_delta_map_local_gaussian(
            labels_fine=tiny_scene["labels_fine"],
            parent_map=tiny_scene["parent_map"],
            images_recon_all=tiny_scene["images_recon_all"],
            label_image=tiny_scene["label_image"],
            sigma=0.1,
            metric="bogus_metric",
        )


def test_local_gaussian_handles_zero_variance():
    """If all recons are identical, scorer should return NaN (degenerate)."""
    H, W = 32, 32
    labels = np.zeros((H, W), dtype=np.int32)
    labels[:H//2, :] = 0
    labels[H//2:, :] = 1
    # 4 identical recons (zero variance)
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    recons = np.stack([img] * 4)

    delta_map, _, labels_used = compute_delta_map_local_gaussian(
        labels_fine=labels,
        parent_map={0: [0], 1: [1]},
        images_recon_all=recons,
        label_image=img,
        sigma=0.1,
        n_realizations=50,
        min_pixels=2,
    )
    assert delta_map.shape == (H, W)
    # Result should be all-NaN since variance is zero
    assert np.all(np.isnan(delta_map))
    assert len(labels_used) == 0
