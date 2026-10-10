"""Local-Gaussian scorer (rohan's theory-based algorithm).

Models p(x) ≈ N(mu, Sigma) locally near x*. Given N reconstructions from q_σ =
N(mu_denoised, Sigma_denoised), recovers the precision matrix Λ via
eigendecomposition of the sample covariance and the closed-form γ→λ relation,
then scores each superpixel by:

    m_i = | 1/2 · ( [Λ(x* − μ)]_i² / Λ_ii − 1 ) |   ("typicality_unsigned")

See rohan/algorithm.tex for derivation, theory_algorithm_rohan.ipynb for the
reference implementation.

Drop-in replacement for ood/scoring.py's compute_delta_map. RGB-only (PCA
features omitted — the derivation assumes the same observation-noise model for
every coordinate, which is true for pixels in [0,1] but not for PCA projections).
"""
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd


SUPPORTED_METRICS = {
    "unnormalized_deviation",
    "normalized_deviation",
    "typicality_signed",
    "typicality_unsigned",
}


def compute_delta_map_local_gaussian(
    labels_fine: np.ndarray,                  # (H, W) int
    parent_map: Dict[int, List[int]],         # unused, kept for signature parity
    images_recon_all: np.ndarray,             # (N, H, W, 3) uint8
    label_image: np.ndarray,                  # (H, W, 3) uint8
    sigma: float,
    n_realizations: int = 1000,
    min_pixels: int = 2,
    seed: int = 0,
    metric: str = "typicality_unsigned",
    variance_explained: float = 0.95,
    eps: float = 1e-12,
    **unused_kwargs,                          # absorb extra eval-pipeline kwargs
) -> Tuple[np.ndarray, Dict, List[int]]:
    """Return (delta_map, info, labels_used).

    Args:
        labels_fine: refined superpixel labels.
        images_recon_all: N reconstructions, uint8 in [0,255].
        label_image: the test/label image, uint8 in [0,255].
        sigma: observation-noise std in [0, 1] image scale (same scale as x after
            dividing by 255). For DDAD at t*=250, ~0.69.
        n_realizations: # random pixel samples for the D-vector covariance.
        min_pixels: skip SPs smaller than this.
        metric: one of SUPPORTED_METRICS.
        seed: RNG seed.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric {metric!r} not in {SUPPORTED_METRICS}")

    H_img, W_img = labels_fine.shape
    assert images_recon_all.ndim == 4 and images_recon_all.shape[-1] == 3
    N = images_recon_all.shape[0]
    C = 3

    images_f = images_recon_all.astype(np.float64) / 255.0
    label_f = label_image.astype(np.float64) / 255.0

    # ---- Select SPs ≥ min_pixels ----
    sp_ids = sorted(np.unique(labels_fine).astype(int).tolist())
    sp_pixel_locs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    valid_sp_ids: List[int] = []
    for sid in sp_ids:
        rows, cols = np.where(labels_fine == sid)
        if len(rows) >= min_pixels:
            sp_pixel_locs[sid] = (rows, cols)
            valid_sp_ids.append(sid)
    n_sp = len(valid_sp_ids)
    if n_sp == 0:
        return np.full((H_img, W_img), np.nan, dtype=np.float32), {}, []

    D = n_sp * C

    # ---- Step 1: (n_realizations, D) matrix by random pixel sampling ----
    rng = np.random.default_rng(seed=seed)
    sp_means = np.zeros((n_realizations, D))
    for k, sid in enumerate(valid_sp_ids):
        rows, cols = sp_pixel_locs[sid]
        n_pix = len(rows)
        img_idx = rng.integers(0, N, size=n_realizations)
        pix_idx = rng.integers(0, n_pix, size=n_realizations)
        vals = images_f[img_idx, rows[pix_idx], cols[pix_idx], :]   # (n_real, C)
        for c in range(C):
            sp_means[:, c * n_sp + k] = vals[:, c]

    # ---- Step 2: SVD-based eigendecomposition of centered X ----
    # Cov(sp_means) eigvecs = right singular vectors of centered X
    # (much cheaper than forming the D×D cov when D >> N).
    X = sp_means - sp_means.mean(axis=0, keepdims=True)
    _, S, Vt = svd(X, full_matrices=False)
    gamma_all = (S ** 2) / (n_realizations - 1)                      # descending

    # Truncate by cumulative variance explained (default 95%).
    # Retaining noise eigenvectors corrupts the gamma→lambda inversion.
    total_var = np.sum(gamma_all)
    if total_var < 1e-15:
        # Degenerate: all samples identical → no signal
        return np.full((H_img, W_img), np.nan, dtype=np.float32), {}, []
    cumvar = np.cumsum(gamma_all) / total_var
    K = max(1, int(np.searchsorted(cumvar, variance_explained) + 1))
    K = min(K, n_realizations - 1)
    gamma = gamma_all[:K]
    U_trunc = Vt[:K, :].T                                            # (D, K)

    # ---- Step 3: gamma -> lambda ----
    alpha = float(sigma) ** (-2)
    lam = ((1 - 2 * alpha * gamma) + np.sqrt(1 + 4 * alpha * gamma)) / (2 * gamma)

    # ---- Step 4: diag(Λ) in full D-space (Λ = U diag(lam) U^T acts on range(U)) ----
    diag_Lambda = np.sum(lam[None, :] * U_trunc ** 2, axis=1)        # (D,)
    safe = diag_Lambda > 1e-15

    # ---- Step 5: empirical mean per SP-channel ----
    mu_denoised = sp_means.mean(axis=0)                              # (D,)

    # ---- Step 6: x_star_sp — label averaged per SP per channel ----
    x_star_sp = np.zeros(D)
    for k, sid in enumerate(valid_sp_ids):
        rows, cols = sp_pixel_locs[sid]
        for c in range(C):
            x_star_sp[c * n_sp + k] = label_f[rows, cols, c].mean()

    # ---- Step 7: mu = (I - B_alpha)^-1 (mu_denoised - B_alpha x_star) ----
    b_diag = alpha / (lam + alpha)
    inv_diag = alpha / lam

    def apply_B(v):
        return U_trunc @ (b_diag * (U_trunc.T @ v))

    def apply_invI_minus_B(v):
        return v + U_trunc @ (inv_diag * (U_trunc.T @ v))

    residual = mu_denoised - apply_B(x_star_sp)
    mu = apply_invI_minus_B(residual)

    # ---- Step 8: metric per SP-channel, then mean over channels ----
    diff = x_star_sp - mu
    Lambda_diff = U_trunc @ (lam * (U_trunc.T @ diff))

    if metric == "unnormalized_deviation":
        m_D = diff ** 2
    else:
        m_D = np.zeros(D)
        m_D[safe] = Lambda_diff[safe] ** 2 / (2 * diag_Lambda[safe])
        if metric == "typicality_signed":
            m_D = m_D - 0.5
        elif metric == "typicality_unsigned":
            m_D = np.abs(m_D - 0.5)

    m_per_sp = m_D.reshape(C, n_sp).T.mean(axis=1)                   # (n_sp,) mean over RGB

    # ---- Step 9: project back to image ----
    delta_map = np.full((H_img, W_img), np.nan, dtype=np.float32)
    info: Dict = {}
    labels_used: List[int] = []
    for k, sid in enumerate(valid_sp_ids):
        rows, cols = sp_pixel_locs[sid]
        delta_map[rows, cols] = m_per_sp[k]
        info[int(sid)] = {"score": float(m_per_sp[k]), "num_pixels": int(len(rows))}
        labels_used.append(int(sid))

    return delta_map, info, sorted(labels_used)
