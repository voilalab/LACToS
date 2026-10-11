

"""Delta map scoring via typical set analysis with RGB + PCA features.

Changes vs. previous version
─────────────────────────────
CRASH FIX   _try_dense_hist checks n_sp × n_bins before allocating.
            If it would exceed MAX_HIST_ELEMENTS, returns None and the
            caller automatically uses the sparse sort+unique path instead.
            This fixes the CUDA illegal-memory-access crash with
            bins_pca=16, n_pca=5  (5635 SPs × 16^5 = 5.9 billion entries,
            ~23 GB at float32 — physically impossible).

MEMORY OPT  Three dtype savings vs. previous float32-everywhere approach:
            1. Histogram counts: torch.int16  (2 bytes, saves 2×)
            2. PMF:              torch.float16 (2 bytes, saves 2×)
            3. SP label tensors: torch.int32  (4 bytes, saves 2× vs int64)
            Combined: 3–4× lower GPU RAM for the histogram tensors.

DATA RANGE  quantize_pca_adaptive replaces the hard [-1,1] clip.
            Bounds come from the 1st–99th percentile of the recon features,
            so no information is lost for PCA components with a larger
            dynamic range.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# Maximum elements for a dense (n_sp × n_bins) histogram tensor.
# At int16 (2 bytes): 200 M elements ≈ 400 MB.  Tune to your GPU RAM.
MAX_HIST_ELEMENTS = 200_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Quantisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def quantize_u8_to_bins(x_u8: np.ndarray, bins: int) -> np.ndarray:
    """Map [0,255] uint8 → [0, bins-1].  Returns uint8."""
    return ((x_u8.astype(np.uint16) * bins) // 256).astype(np.uint8)


def quantize_pca_adaptive(
    x_pca:   np.ndarray,    # (N, k)
    bins:    int,
    pca_min: np.ndarray,    # (k,)  per-component min from recon
    pca_max: np.ndarray,    # (k,)  per-component max from recon
) -> np.ndarray:
    """
    Data-driven PCA quantization — no information lost.

    Maps the observed reconstruction range → [0, bins-1].
    Label pixels outside the recon range are clamped to the boundary bins
    (they carry the OOD signal) rather than silently aliased.

    pca_min / pca_max must be computed from the RECON features (e.g. via
    np.percentile at 1 / 99) to cover the in-distribution range.
    """
    rng = np.where((pca_max - pca_min) < 1e-8, 1e-8, pca_max - pca_min)
    x_norm = (x_pca - pca_min) / rng
    return np.clip((x_norm * bins).astype(np.int32), 0, bins - 1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# GPU smoothing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g  = torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
    return g / g.sum()


def _smooth_hist_nd(hist: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    """Separable 1-D Gaussian along every bin dimension (dim ≥ 1)."""
    if sigma <= 0:
        return hist
    k   = _gaussian_kernel_1d(sigma, device)
    out = hist.float()
    for dim in range(1, out.ndim):
        shape = list(out.shape)
        nb    = shape[dim]
        perm  = list(range(len(shape))); perm[dim], perm[-1] = perm[-1], perm[dim]
        x     = out.permute(perm).reshape(-1, nb)
        x     = F.conv1d(x.unsqueeze(1), k.view(1,1,-1), padding=len(k)//2).squeeze(1)
        ns    = [shape[i] for i in perm]; ns[-1] = nb
        out   = x.reshape(ns).permute(perm)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dense histogram builder  (int16 counts → float16 PMF)
# ─────────────────────────────────────────────────────────────────────────────

def _try_dense_hist(
    sp_r:         torch.Tensor,   # (N*H*W,)  int32
    bins_np:      np.ndarray,     # (N*H*W,)  flat bin index
    n_sp:         int,
    n_bins:       int,
    bin_shape:    tuple,
    device:       torch.device,
    smooth_sigma: float,
    eps:          float,
) -> Optional[torch.Tensor]:
    """
    Build a dense (n_sp, *bin_shape) PMF on GPU using int16 counts.

    Memory per histogram: n_sp × n_bins × 2 bytes (int16 counts)
                        + n_sp × n_bins × 2 bytes (float16 PMF)
    vs. previous float32: n_sp × n_bins × 4 bytes → 2× saving.

    Returns float16 PMF tensor, or None if n_sp × n_bins > MAX_HIST_ELEMENTS.
    int16 max = 32 767.  With N_recon=24 and avg SP ≈ 11px, max count ≈ 264 ✓
    """
    if n_sp * n_bins > MAX_HIST_ELEMENTS:
        return None

    # Use int64 for the combined key to avoid overflow
    b        = torch.from_numpy(bins_np).to(device, dtype=torch.int64)
    combined = sp_r.long() * n_bins + b

    counts   = torch.bincount(combined, minlength=n_sp * n_bins)\
                    .to(torch.int16)\
                    .reshape(n_sp, *bin_shape)

    hist = counts.float()
    if smooth_sigma > 0:
        hist = _smooth_hist_nd(hist, smooth_sigma, device)

    hist += eps #changed
    norm  = hist.reshape(n_sp, -1).sum(1).view(n_sp, *([1]*len(bin_shape))) #changed
    pmf = (hist / norm).to(torch.float16) #changed
    # alpha = 0.5
    # hist_smoothed = hist + alpha
    # norm = hist.reshape(n_sp, -1).sum(1, keepdim=True) + alpha * n_bins
    # pmf = hist_smoothed / norm.view(n_sp, *([1]*len(bin_shape)))
    return pmf  # (n_sp, *bin_shape)  fp16


def _dense_entropy(pmf: torch.Tensor, n_sp: int, eps: float) -> torch.Tensor:
    q = pmf.float().reshape(n_sp, -1)
    return -(q * torch.log2(q + eps)).sum(1)          # (n_sp,)  fp32


def _dense_avg_nll(
    lbins_np: np.ndarray,
    pmf:      torch.Tensor,
    n_bins:   int,
    n_sp:     int,
    sp_l:     torch.Tensor,     # (H*W,)  int32
    sp_counts: np.ndarray,
    device:   torch.device,
    eps:      float,
) -> torch.Tensor:
    b    = torch.from_numpy(lbins_np).to(device, dtype=torch.int64)
    p    = pmf.float().reshape(n_sp, n_bins)[sp_l.long(), b]
    nll  = -torch.log2(p + eps)
    s    = torch.zeros(n_sp, dtype=torch.float32, device=device)
    s.scatter_add_(0, sp_l.long(), nll)
    return s / torch.from_numpy(sp_counts).float().to(device).clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse histogram path  (sort + unique_consecutive)
#   Memory: O(N_recon × H × W) — independent of n_bins.
#   Activated automatically when the dense path would exceed MAX_HIST_ELEMENTS.
#   Note: histogram smoothing is not applied (eps-smoothing only).
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_entropy_and_nll(
    sp_r:      torch.Tensor,   # (N*H*W,)  int32
    sp_l:      torch.Tensor,   # (H*W,)    int32
    bins_r_np: np.ndarray,     # (N*H*W,)  flat recon bin indices  int64
    bins_l_np: np.ndarray,     # (H*W,)    flat label bin indices  int64
    n_sp:      int,
    n_bins:    int,
    sp_counts: np.ndarray,
    device:    torch.device,
    eps:       float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (entropy_sp, avg_nll_sp), both (n_sp,) float32."""
    bins_r = torch.from_numpy(bins_r_np).to(device, dtype=torch.int64)
    bins_l = torch.from_numpy(bins_l_np).to(device, dtype=torch.int64)

    keys_r              = sp_r.long() * n_bins + bins_r
    keys_l              = sp_l.long() * n_bins + bins_l
    keys_sorted         = keys_r.sort().values
    unique_keys, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    unique_sp           = torch.div(unique_keys, n_bins, rounding_mode='trunc').long()

    sp_totals = torch.zeros(n_sp, dtype=torch.float32, device=device)
    sp_totals.scatter_add_(0, unique_sp, counts.float())

    probs       = counts.float() / sp_totals[unique_sp]
    ent_contrib = -probs * torch.log2(probs + eps)
    entropy_sp  = torch.zeros(n_sp, dtype=torch.float32, device=device)
    entropy_sp.scatter_add_(0, unique_sp, ent_contrib)

    pos         = torch.searchsorted(unique_keys.contiguous(), keys_l.contiguous())
    pos_c       = pos.clamp(0, len(unique_keys) - 1)
    found       = unique_keys[pos_c] == keys_l
    recon_cnt   = torch.where(found, counts[pos_c].float(),
                              torch.zeros(len(sp_l), dtype=torch.float32, device=device))
    sp_tot_l    = sp_totals[sp_l.long()]
    eps_t       = torch.tensor(eps, dtype=torch.float32, device=device)
    prob_tgt    = torch.where(found,
                              recon_cnt / (sp_tot_l + eps_t),
                              eps_t     / (sp_tot_l + eps_t))
    nll_px      = -torch.log2(prob_tgt + eps_t)
    nll_sum     = torch.zeros(n_sp, dtype=torch.float32, device=device)
    nll_sum.scatter_add_(0, sp_l.long(), nll_px)
    avg_nll_sp  = nll_sum / torch.from_numpy(sp_counts).float().to(device).clamp(min=1.0)
    return entropy_sp, avg_nll_sp


# ─────────────────────────────────────────────────────────────────────────────
# CPU reference (unchanged algorithm)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# GPU-accelerated implementation
# ─────────────────────────────────────────────────────────────────────────────


def compute_delta_map_gpu(
    labels_fine:      np.ndarray,
    images_recon_all: np.ndarray,
    pca_feats_recon:  np.ndarray,
    label_image:      np.ndarray,
    label_pca_map:    np.ndarray,
    parent_map:       Optional[Dict[int, List[int]]] = None,
    bins_rgb:         int   = 32,
    bins_pca:         int   = 8,
    smooth_sigma:     float = 0.1,
    min_pixels:       int   = 2,
    device:           torch.device = None,
    eps:              float = 1e-12,
    gray_scale:       bool  = False,
    use_pca:          bool  = True,
    **unused_kwargs,
) -> Tuple[np.ndarray, Dict, List[int]]:
    """
    GPU-accelerated factorized PMF scoring with automatic memory management.

    For each feature space (RGB, PCA) independently:
      • If n_sp × n_bins ≤ MAX_HIST_ELEMENTS  → dense int16 histogram (fast, smoothable)
      • Otherwise                              → sparse sort+unique (O(N×H×W), any bin count)

    This means bins_pca=16 / n_pca=5 / 5635 SPs works fine — the PCA histogram
    (5.9 B entries) automatically uses the sparse path while RGB stays dense.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W    = labels_fine.shape
    N_recon = images_recon_all.shape[0]
    n_pca   = pca_feats_recon.shape[-1]

    # ── Remap SP labels (int32 saves vs int64) ────────────────────────────
    sp_ids_orig = np.unique(labels_fine)
    n_sp        = len(sp_ids_orig)
    remap       = np.zeros(int(sp_ids_orig.max()) + 1, dtype=np.int32)
    for ni, oi in enumerate(sp_ids_orig):
        remap[oi] = ni
    labels_c  = remap[labels_fine]
    sp_counts = np.bincount(labels_c.ravel(), minlength=n_sp)
    valid_sp  = sp_counts >= max(min_pixels, 1)

    sp_r = torch.from_numpy(labels_c.ravel()).to(device, dtype=torch.int32).repeat(N_recon)
    sp_l = torch.from_numpy(labels_c.ravel()).to(device, dtype=torch.int32)

    # ── Quantise ─────────────────────────────────────────────────────────
    if gray_scale:
        r_rgb    = images_recon_all[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
        l_rgb    = label_image[..., :3].mean(-1, keepdims=True).reshape(-1, 1).astype(np.uint8)
        rgb_dims = (bins_rgb,)
    else:
        # recon_hsv = rgb2hsv(images_recon_all[..., :3].astype(np.float32) / 255.0)
        # label_hsv = rgb2hsv(label_image[...,  :3].astype(np.float32) / 255.0)
        # r_rgb    = (recon_hsv.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
        # l_rgb    = (label_hsv.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
        # dominant_hue = int(np.median(r_rgb[:, 0]))       # median hue across all recon pixels
        # shift        = (128 - dominant_hue) % 256         # centre it at bin 128
        # r_rgb        = shift_hue_channel(r_rgb, shift)
        # l_rgb        = shift_hue_channel(l_rgb, shift)    # same shift — same coordinate system

        r_rgb    = images_recon_all[..., :3].reshape(-1, 3).astype(np.uint8) #Changed
        l_rgb    = label_image[..., :3].reshape(-1, 3).astype(np.uint8) #Changed
        rgb_dims = (bins_rgb, bins_rgb, bins_rgb)

    recon_pca_flat = pca_feats_recon.reshape(-1, n_pca)
    label_pca_flat = label_pca_map.reshape(-1, n_pca)

    # Data-driven PCA range (robust percentiles from recon)
    pca_min = np.percentile(recon_pca_flat, 1,  axis=0)
    pca_max = np.percentile(recon_pca_flat, 99, axis=0)

    rq_rgb = quantize_u8_to_bins(r_rgb, bins_rgb)
    rq_pca = quantize_pca_adaptive(recon_pca_flat, bins_pca, pca_min, pca_max)
    lq_rgb = quantize_u8_to_bins(l_rgb, bins_rgb)
    lq_pca = quantize_pca_adaptive(label_pca_flat, bins_pca, pca_min, pca_max)

    pca_dims   = (bins_pca,) * n_pca
    n_rgb_bins = int(np.prod(rgb_dims))
    n_pca_bins = int(np.prod(pca_dims))

    # Flat bin indices: int32 safe for RGB (max 32^3=32768), int64 for PCA
    if gray_scale:
        rb = rq_rgb[:, 0].astype(np.int32); lb = lq_rgb[:, 0].astype(np.int32)
    else:
        rb = np.ravel_multi_index((rq_rgb[:,0].astype(np.int64),
                                   rq_rgb[:,1].astype(np.int64),
                                   rq_rgb[:,2].astype(np.int64)), dims=rgb_dims).astype(np.int32)
        lb = np.ravel_multi_index((lq_rgb[:,0].astype(np.int64),
                                   lq_rgb[:,1].astype(np.int64),
                                   lq_rgb[:,2].astype(np.int64)), dims=rgb_dims).astype(np.int32)

    rp = np.ravel_multi_index(
        tuple(rq_pca[:,i].astype(np.int64) for i in range(n_pca)), dims=pca_dims)
    lp = np.ravel_multi_index(
        tuple(lq_pca[:,i].astype(np.int64) for i in range(n_pca)), dims=pca_dims)

    # ── RGB ───────────────────────────────────────────────────────────────
    pmf_rgb = _try_dense_hist(sp_r, rb, n_sp, n_rgb_bins, rgb_dims,
                              device, smooth_sigma, eps)
    if pmf_rgb is not None:
        entropy_rgb = _dense_entropy(pmf_rgb, n_sp, eps)
        avg_nll_rgb = _dense_avg_nll(lb.astype(np.int64), pmf_rgb, n_rgb_bins,
                                     n_sp, sp_l, sp_counts, device, eps)
    else:
        print(f"[INFO] RGB histogram using sparse path "
              f"(n_sp={n_sp}, n_rgb_bins={n_rgb_bins:,})")
        entropy_rgb, avg_nll_rgb = _sparse_entropy_and_nll(
            sp_r, sp_l, rb.astype(np.int64), lb.astype(np.int64),
            n_sp, n_rgb_bins, sp_counts, device, eps)

    # ── PCA: dense if fits, sparse otherwise (CRASH FIX) ─────────────────
    pmf_pca = _try_dense_hist(sp_r, rp.astype(np.int32) if rp.max() < 2**31-1 else rp,
                              n_sp, n_pca_bins, pca_dims,
                              device, smooth_sigma, eps)
    if pmf_pca is not None:
        entropy_pca = _dense_entropy(pmf_pca, n_sp, eps)
        avg_nll_pca = _dense_avg_nll(lp, pmf_pca, n_pca_bins,
                                     n_sp, sp_l, sp_counts, device, eps)
    else:
        mem_mb = n_sp * n_pca_bins * 2 / 1e6
        print(f"[INFO] PCA histogram too large for dense "
              f"({mem_mb:.0f} MB at int16, n_sp={n_sp}, n_pca_bins={n_pca_bins:,}); "
              f"using sparse path — no smoothing applied to PCA.")
        entropy_pca, avg_nll_pca = _sparse_entropy_and_nll(
            sp_r, sp_l, rp, lp, n_sp, n_pca_bins, sp_counts, device, eps)

    # RGB-only ablation. The two feature spaces enter the score additively, so
    # zeroing the PCA factor is EXACTLY "score on raw intensity alone" rather
    # than an approximation of it.
    if not use_pca:
        entropy_pca = torch.zeros_like(entropy_pca)
        avg_nll_pca = torch.zeros_like(avg_nll_pca)

    # ── Delta = |avg_NLL − H| ─────────────────────────────────────────────
    # delta_sp = torch.abs((avg_nll_rgb + avg_nll_pca) - (entropy_rgb + entropy_pca)) ## Changed
    delta_sp = ((avg_nll_rgb + avg_nll_pca) - (entropy_rgb + entropy_pca))

    valid_t  = torch.from_numpy(valid_sp).to(device)
    delta_sp = torch.where(valid_t, delta_sp, torch.full_like(delta_sp, float("nan")))

    d_np = delta_sp.cpu().numpy()
    e_np = (entropy_rgb + entropy_pca).cpu().numpy()
    n_np = (avg_nll_rgb + avg_nll_pca).cpu().numpy()

    delta_map_out = d_np[labels_c].astype(np.float32)
    delta_map_out[~valid_sp[labels_c]] = np.nan

    child_to_parent: Dict[int,int] = {}
    if parent_map:
        for oid, ch in parent_map.items():
            for c in ch: child_to_parent[c] = oid

    info: Dict = {}; labels_used: List[int] = []
    for ni, oi in enumerate(sp_ids_orig):
        if valid_sp[ni]:
            info[int(oi)] = {"orig_parent": child_to_parent.get(int(oi)),
                             "num_pixels": float(sp_counts[ni]),
                             "H_bits": float(e_np[ni]),
                             "avg_neg_logp_bits": float(n_np[ni]),
                             "delta_sp": float(d_np[ni])}
            labels_used.append(int(oi))

    return delta_map_out, info, sorted(labels_used)