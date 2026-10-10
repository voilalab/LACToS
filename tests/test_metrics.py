"""Tests for ood/metrics.py — AUROC / AP computation."""
import numpy as np
import pytest

from ood.metrics import evaluate_delta_map


def test_perfect_separation_gives_auc_1():
    """Delta map matches GT exactly → AUC = 1.0."""
    H, W = 16, 16
    labels_fine = np.zeros((H, W), dtype=np.int32)
    # 4 SPs: top-left, top-right, bottom-left, bottom-right
    labels_fine[:H//2, :W//2] = 0
    labels_fine[:H//2, W//2:] = 1
    labels_fine[H//2:, :W//2] = 2
    labels_fine[H//2:, W//2:] = 3

    # GT: only SPs 0 and 2 are anomalous
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    gt_mask[labels_fine == 0] = 1
    gt_mask[labels_fine == 2] = 1

    # Delta: high for anomalous SPs, low for normal
    delta_map = np.zeros((H, W), dtype=np.float32)
    delta_map[labels_fine == 0] = 10.0
    delta_map[labels_fine == 1] = 0.1
    delta_map[labels_fine == 2] = 10.0
    delta_map[labels_fine == 3] = 0.1

    result = evaluate_delta_map(
        delta_map=delta_map, labels_fine=labels_fine,
        gt_mask_binary=gt_mask, anomaly_threshold=0.5,
        smooth_sigma=None,
    )
    assert result["sp_roc_auc"] == pytest.approx(1.0)
    assert result["px_roc_auc"] == pytest.approx(1.0)


def test_inverted_delta_gives_auc_0():
    """Delta inverted relative to GT → AUC = 0.0 (anti-discriminative)."""
    H, W = 16, 16
    labels_fine = np.zeros((H, W), dtype=np.int32)
    labels_fine[:, :W//2] = 0
    labels_fine[:, W//2:] = 1

    gt_mask = np.zeros((H, W), dtype=np.uint8)
    gt_mask[labels_fine == 0] = 1  # SP 0 is anomalous

    delta_map = np.zeros((H, W), dtype=np.float32)
    delta_map[labels_fine == 0] = 0.1   # LOW on anomalous
    delta_map[labels_fine == 1] = 10.0  # HIGH on normal

    result = evaluate_delta_map(
        delta_map=delta_map, labels_fine=labels_fine,
        gt_mask_binary=gt_mask, anomaly_threshold=0.5,
        smooth_sigma=None,
    )
    assert result["sp_roc_auc"] == pytest.approx(0.0)


def test_random_delta_near_05():
    """Random delta values should give AUROC ≈ 0.5 within noise tolerance."""
    rng = np.random.default_rng(0)
    H, W = 32, 32
    # Make 16 SPs
    labels_fine = np.zeros((H, W), dtype=np.int32)
    for i in range(4):
        for j in range(4):
            labels_fine[i*8:(i+1)*8, j*8:(j+1)*8] = i * 4 + j
    # Half anomalous
    anomalous_sps = rng.choice(16, size=8, replace=False)
    gt_mask = np.isin(labels_fine, anomalous_sps).astype(np.uint8)
    # Random delta per SP
    delta_map = np.zeros((H, W), dtype=np.float32)
    for sid in range(16):
        delta_map[labels_fine == sid] = rng.random()
    result = evaluate_delta_map(
        delta_map=delta_map, labels_fine=labels_fine,
        gt_mask_binary=gt_mask, anomaly_threshold=0.5,
        smooth_sigma=None,
    )
    # With only 16 SPs the variance is high — allow a generous tolerance
    assert 0.25 <= result["sp_roc_auc"] <= 0.75


def test_evaluate_returns_expected_keys():
    H, W = 8, 8
    labels_fine = np.zeros((H, W), dtype=np.int32)
    labels_fine[:, :W//2] = 0
    labels_fine[:, W//2:] = 1
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[labels_fine == 0] = 1
    delta = np.where(labels_fine == 0, 10.0, 0.1).astype(np.float32)
    result = evaluate_delta_map(
        delta_map=delta, labels_fine=labels_fine,
        gt_mask_binary=gt, anomaly_threshold=0.5,
        smooth_sigma=None,
    )
    for key in ["sp_roc_auc", "px_roc_auc", "sp_ap", "px_ap"]:
        assert key in result
