from skimage import exposure, io
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import gaussian, sobel
from skimage.segmentation import find_boundaries, slic, watershed
from skimage.transform import resize
from skimage.util import img_as_float

def _robust_rescale(x, p_low=1.0, p_high=99.0):
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32) 

def _build_edge_map(img, edge_sigma=1.0, clahe_clip=0.02, alpha_grad=10.0):
    f = img_as_float(img).astype(np.float32)
    gray = f if f.ndim == 2 else rgb2gray(f).astype(np.float32)
    gray = _robust_rescale(gray)
    gray_eq = exposure.equalize_adapthist(gray, clip_limit=clahe_clip).astype(np.float32)
    gs  = gaussian(gray,    sigma=edge_sigma, preserve_range=True).astype(np.float32)
    gse = gaussian(gray_eq, sigma=edge_sigma, preserve_range=True).astype(np.float32)
    grad = _robust_rescale(0.35 * sobel(gs) + 0.65 * sobel(gse))
    return np.power(grad, max(1.0, alpha_grad / 5.0)).astype(np.float32)


def _make_grid_markers(shape, target_size):
    H, W  = shape
    step  = max(2, int(np.sqrt(max(1, target_size))))
    m     = np.zeros((H, W), dtype=np.int32)
    label = 1
    for r in range(step//2, H, step):
        for c in range(step//2, W, step):
            m[r, c] = label; label += 1
    return m

def recursive_subdivide(img, labels, var_threshold=150.0, min_pixels=16,
                         max_sub=6, max_depth=4, compactness=1.0,
                         alpha_grad=1.0, target_size=300):
    edge      = _build_edge_map(img, alpha_grad=alpha_grad)
    markers   = _make_grid_markers(edge.shape, target_size)
    labels_ws = watershed(edge, markers, compactness=max(0.0, compactness)/1000.0)
    uniq      = np.unique(labels_ws)
    remap2    = np.zeros(uniq.max()+1, dtype=np.int32)
    for i, v in enumerate(uniq):
        remap2[v] = i
    labels_out = remap2[labels_ws]
    final_ids  = sorted(labels_out.ravel().tolist())
    parent_map = {int(o): np.unique(labels_out[labels == o]).tolist()
                  for o in np.unique(labels)}
    return labels_out, final_ids, parent_map


"""Recursive color-homogeneous superpixel refinement.

Extracted from New_dataset_clean.ipynb Cell 5. This is the analysis-side
superpixel logic (NOT super_pixel_generation.py, which is for sampling masks).

No changes vs. original — this file is the reference implementation that
full_run_hist.py was corrected to match.
"""

from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.util import img_as_float


def rgb_variance(img: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-channel variance of RGB inside mask."""
    if mask.sum() == 0:
        return 0.0
    pix = img[mask]
    if pix.ndim == 1:
        return float(np.var(pix))
    return float(np.mean([np.var(pix[:, c]) for c in range(pix.shape[1])]))


def slic_features_lab(img: np.ndarray, alpha_grad: float = 10.0) -> np.ndarray:
    """SLIC features: Lab color + gradient edge channel."""
    img_f = img_as_float(img).astype(np.float32)
    if img_f.ndim == 2:
        g = sobel(img_f)
        return np.dstack([img_f, alpha_grad * g])
    lab = rgb2lab(img_f).astype(np.float32)
    gray = rgb2gray(img_f)
    g = sobel(gray).astype(np.float32)
    return np.dstack([lab, alpha_grad * g[..., None]])


def split_region(
    img: np.ndarray,
    full_mask: np.ndarray,
    region_id: int,
    n_sub: int = 4,
    compactness: float = 8.0,
    alpha_grad: float = 10.0,
) -> np.ndarray:
    """Split region_id in full_mask into n_sub sub-regions via SLIC."""
    m = full_mask == region_id
    if m.sum() < n_sub:
        return full_mask

    rows, cols = np.where(m)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    img_crop = img[r0:r1, c0:c1]
    m_crop = m[r0:r1, c0:c1]

    feats = slic_features_lab(img_crop, alpha_grad=alpha_grad)

    sub = slic(
        feats,
        n_segments=n_sub,
        compactness=compactness,
        sigma=0.5,
        start_label=0,
        mask=m_crop,
        channel_axis=-1,
    )

    new_labels = full_mask.copy()
    base = int(full_mask.max()) + 1
    rr, cc = np.where(m_crop)
    used = np.unique(sub[rr, cc])
    used = used[used >= 0]

    for i, u in enumerate(used):
        sel = (sub == u) & m_crop
        r_sel, c_sel = np.where(sel)
        new_labels[r0 + r_sel, c0 + c_sel] = base + i

    return new_labels


# def recursive_subdivide(
#     img: np.ndarray,
#     labels: np.ndarray,
#     var_threshold: float = 150.0,
#     min_pixels: int = 16,
#     max_sub: int = 6,
#     max_depth: int = 4,
#     compactness: float = 1.0,
#     alpha_grad: float = 1.0,
#     target_size: int = 100,
# ) -> Tuple[np.ndarray, List[int], Dict[int, List[int]]]:
#     """Recursively split superpixels until color-homogeneous.

#     Returns:
#         labels_out: (H, W) refined label map.
#         final_ids: sorted list of all label IDs.
#         parent_map: original_id -> [child_ids].
#     """
#     labels_out = labels.copy().astype(np.int32)

#     orig_ids = list(np.unique(labels_out))
#     parent_map: Dict[int, List[int]] = {int(oid): [int(oid)] for oid in orig_ids}

#     def _update_parent(orig_parent: int, old_child: int, new_children: List[int]) -> None:
#         children = parent_map[int(orig_parent)]
#         children = [c for c in children if c != old_child]
#         children.extend(new_children)
#         parent_map[int(orig_parent)] = children

#     queue: deque = deque()
#     for oid in orig_ids:
#         queue.append((int(oid), 0, int(oid)))

#     while queue:
#         rid, depth, orig_parent = queue.popleft()

#         m = labels_out == rid
#         area = int(m.sum())

#         if area < min_pixels or depth >= max_depth:
#             continue

#         v = rgb_variance(img, m)
#         if v < var_threshold:
#             continue

#         n_sub = max(2, min(max_sub, area // target_size))
#         if n_sub < 2:
#             continue

#         old_max = int(labels_out.max())
#         labels_out = split_region(
#             img, labels_out, rid,
#             n_sub=n_sub,
#             compactness=compactness,
#             alpha_grad=alpha_grad,
#         )
#         new_max = int(labels_out.max())

#         if new_max == old_max:
#             continue

#         new_children = list(range(old_max + 1, new_max + 1))

#         residual_mask = labels_out == rid
#         if residual_mask.sum() > 0:
#             residual_id = int(labels_out.max()) + 1
#             labels_out[residual_mask] = residual_id
#             new_children.append(residual_id)

#         _update_parent(orig_parent, rid, new_children)

#         for child in new_children:
#             queue.append((child, depth + 1, orig_parent))

#     final_ids = sorted(np.unique(labels_out).astype(int).tolist())
#     return labels_out, final_ids, parent_map


def pool_image(
    img: np.ndarray, labels: np.ndarray, reduce: str = "mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace every pixel with its region's mean/median value."""
    lab = labels.astype(np.int32)
    K = int(lab.max()) + 1

    if img.ndim == 2:
        flat_lab = lab.ravel()
        flat_x = img.ravel().astype(np.float64)
        if reduce == "mean":
            sums = np.bincount(flat_lab, weights=flat_x, minlength=K)
            counts = np.maximum(np.bincount(flat_lab, minlength=K).astype(np.float64), 1.0)
            vals = (sums / counts).astype(np.float32)
        else:
            vals = np.zeros(K, dtype=np.float32)
            for k in range(K):
                m = lab == k
                if m.any():
                    vals[k] = np.median(img[m])
        return vals[lab], vals

    C = img.shape[-1]
    vals = np.zeros((K, C), dtype=np.float32)
    flat_lab = lab.ravel()
    counts = np.maximum(np.bincount(flat_lab, minlength=K).astype(np.float64), 1.0)

    if reduce == "mean":
        for c in range(C):
            sums = np.bincount(flat_lab, weights=img[..., c].ravel().astype(np.float64), minlength=K)
            vals[:, c] = (sums / counts).astype(np.float32)
    else:
        for k in range(K):
            m = lab == k
            if not m.any():
                continue
            for c in range(C):
                vals[k, c] = np.median(img[..., c][m])

    return vals[lab], vals
