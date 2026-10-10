"""Evaluation metrics: ROC, PR, AUC — pure numpy, no sklearn.

Changes vs. original (aligned with full_run_hist.py):
  FIX 1  manual_average_precision: sign was wrong (-np.sum → +np.sum).
          The original produced NEGATIVE AP values for every sample.
  FIX 2  manual_auc: use getattr(np, "trapezoid", np.trapz) for numpy 2.x
          compatibility (np.trapz is deprecated in numpy >= 2.0).
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def manual_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FPR, TPR for all thresholds."""
    desc_idx = np.argsort(-y_score)
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    distinct_idx = np.where(np.diff(y_score))[0]
    threshold_idx = np.concatenate([distinct_idx, [len(y_true) - 1]])

    tps = np.cumsum(y_true)[threshold_idx]
    fps = (threshold_idx + 1) - tps

    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])

    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    tpr = tps / tps[-1] if tps[-1] > 0 else tps

    thresholds = y_score[threshold_idx]
    return fpr, tpr, thresholds


def manual_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC.

    FIX 2: np.trapz is deprecated in numpy >= 2.0.  Use np.trapezoid when
    available, fall back to np.trapz on older versions.
    """
    order = np.argsort(x, kind="stable")  # FIX 3: stable sort preserves TPR order within tied FPR runs
    x, y = x[order], y[order]
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x))


def manual_precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall for all thresholds.

    FIX 4: group tied scores onto a single threshold instead of walking one
    sample at a time. `manual_auc` already needed a stable sort because an
    unstable sort permuted TPR values within tied-FPR runs; this function had
    the same class of defect one level up — it neither sorted stably nor
    grouped ties, so within a block of equal scores, precision/recall were
    evaluated once per sample in whatever order argsort happened to place
    same-valued entries, instead of once per distinct threshold (as
    sklearn's precision_recall_curve does). That biases AP low, worse on
    maps with many tied/quantized scores (e.g. a blocky superpixel delta
    map) than on maps with mostly-distinct scores. Mirrors the distinct-value
    grouping `manual_roc_curve` already does.
    """
    desc_idx = np.argsort(-y_score, kind="stable")
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    distinct_idx = np.where(np.diff(y_score))[0]
    threshold_idx = np.concatenate([distinct_idx, [len(y_true) - 1]])

    tps = np.cumsum(y_true)[threshold_idx]
    fps = (threshold_idx + 1) - tps
    total_pos = y_true.sum()

    precision = tps / (tps + fps)
    recall = tps / total_pos if total_pos > 0 else tps

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    thresholds = y_score[threshold_idx]

    return precision, recall, thresholds


def manual_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under PR curve).

    FIX 1: the original used -np.sum(np.diff(recall) * precision[:-1]) which
    always produced negative values.  The correct formula is +np.sum(...).
    FIX 5: precision[:-1] paired each recall increment with the precision
    *before* the step (the left/lower-recall endpoint of the interval).
    The standard definition, AP = sum_n (R_n - R_n-1) * P_n (and sklearn's
    average_precision_score), pairs each increment with the precision *at*
    the step it produced — precision[1:], the right endpoint — since
    `manual_precision_recall_curve`'s arrays are prepended with the
    threshold=+inf point (precision=1.0, recall=0.0), not appended.  Verified
    exact (to float64 rounding, ~1e-16) against sklearn across many seeds,
    including score arrays with heavy ties, once combined with the FIX 4
    tie-grouping in manual_precision_recall_curve.
    """
    precision, recall, _ = manual_precision_recall_curve(y_true, y_score)
    return float(np.sum(np.diff(recall) * precision[1:]))


def evaluate_delta_map(
    delta_map: np.ndarray,
    labels_fine: np.ndarray,
    gt_mask_binary: np.ndarray,
    anomaly_threshold: float = 0.5,
    smooth_sigma: Optional[float] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Run both superpixel-level and pixel-level evaluation.

    Args:
        delta_map: (H, W) float array of anomaly scores (may contain NaN).
        labels_fine: (H, W) int array of refined superpixel labels.
        gt_mask_binary: (H, W) uint8 binary ground-truth mask.
        anomaly_threshold: fraction of anomalous pixels for SP majority vote.
        smooth_sigma: if not None, apply Gaussian smoothing to delta_map first.
        valid_mask: (H, W) boolean array indicating valid evaluation region.

    Returns:
        dict with sp_roc_auc, sp_ap, px_roc_auc, px_ap, and curve data.
    """
    # Superpixel-level uses raw (NaN-safe) scores
    score_map_sp = np.nan_to_num(delta_map, nan=0.0)

    # Pixel-level: optionally smooth delta_map
    if smooth_sigma is not None:
        score_map_px = gaussian_filter(np.nan_to_num(delta_map, nan=0.0), sigma=smooth_sigma, mode="nearest")
    else:
        score_map_px = delta_map.copy()

    # --- Superpixel-level ---
    all_sp_ids = sorted(np.unique(labels_fine).astype(int).tolist())
    sp_scores = []
    sp_labels = []

    for sp_id in all_sp_ids:
        sp_mask = labels_fine == sp_id
        n_pix = int(sp_mask.sum())
        if n_pix < 1:
            continue

        score = float(np.nanmean(score_map_sp[sp_mask]))
        if np.isnan(score):
            continue

        frac_anomalous = gt_mask_binary[sp_mask].mean()
        label = int(frac_anomalous > anomaly_threshold)

        sp_scores.append(score)
        sp_labels.append(label)

    sp_scores = np.array(sp_scores)
    sp_labels = np.array(sp_labels)

    fpr_sp, tpr_sp, thresh_sp = manual_roc_curve(sp_labels, sp_scores)
    sp_roc_auc = manual_auc(fpr_sp, tpr_sp)
    sp_ap = manual_average_precision(sp_labels, sp_scores)

    # --- Pixel-level ---
    if valid_mask is not None:
        valid = (~np.isnan(score_map_px)) & valid_mask.astype(bool)
    else:
        valid = ~np.isnan(score_map_px)
        
    pixel_scores = score_map_px[valid].ravel()
    pixel_labels = gt_mask_binary[valid].ravel()

    fpr_px, tpr_px, thresh_px = manual_roc_curve(pixel_labels, pixel_scores)
    px_roc_auc = manual_auc(fpr_px, tpr_px)
    px_ap = manual_average_precision(pixel_labels, pixel_scores)

    ood_v = score_map_px[gt_mask_binary.astype(bool) & valid]
    id_v  = score_map_px[~gt_mask_binary.astype(bool) & valid]
    if len(ood_v) > 0 and len(id_v) > 0:
        snr = float((np.mean(ood_v) - np.mean(id_v)) / (np.std(id_v) + 1e-8))
    else:
        snr = 0.0
    score_min = score_map_px[valid].min()
    score_max = score_map_px[valid].max()
    score_norm = (score_map_px - score_min) / (score_max - score_min + 1e-8)

    ood_pixels = score_norm[gt_mask_binary.astype(bool) & valid]   # anomalous pixels
    id_pixels  = score_norm[~gt_mask_binary.astype(bool) & valid]  # normal pixels

    if len(ood_pixels) > 0 and len(id_pixels) > 0:
        # 2. MSE = squared difference between mean OOD score and mean ID score
        #    (signal = mean anomaly score, noise = mean normal score)
        mu_ood = float(np.mean(ood_pixels))
        mu_id  = float(np.mean(id_pixels))
        mse    = (mu_ood - mu_id) ** 2

        # 3. MAX_I = 1 (after global normalization)
        # PSNR = 10 * log10(MAX_I^2 / MSE) = 10 * log10(1 / MSE)
        if mse > 0:
            psnr = float(10 * np.log10(1.0 / mse))
        else:
            psnr = float("inf")
    else:
        # No pixels of one class (e.g. an anomaly-free image): leave mse
        # undefined instead of falling through to the block below, which used
        # to raise UnboundLocalError on `mse`.
        mse  = float("nan")
        psnr = float("nan")
    if not np.isfinite(mse):
        psnr = float("nan")
    elif mse > 0:
        psnr = float(10 * np.log10(1.0 / mse))
        psnr = min(psnr, 100.0)   # cap at 100 dB — avoids inf formatting issues
    else:
        psnr = 100.0
    return {
        "sp_roc_auc": sp_roc_auc,
        "sp_ap": sp_ap,
        "px_roc_auc": px_roc_auc,
        "px_ap": px_ap,
        "snr": snr,
        "psnr": psnr,
        "num_superpixels": len(sp_scores),
        "num_anomalous_sp": int(sp_labels.sum()),
        "curves": {
            "sp_fpr": fpr_sp.tolist(),
            "sp_tpr": tpr_sp.tolist(),
            "px_fpr": fpr_px.tolist(),
            "px_tpr": tpr_px.tolist(),
        },
    }
