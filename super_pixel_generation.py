"""Generate an initial SLIC superpixel mask + boundary visualization.

CLI tool called by `evaluate.py` (and `scripts/run_faces.sh`) to produce
`mask.png` for a test image. The mask is the starting point for the recursive
refinement in `ood/superpixels.py`.

Pipeline:
  1. Load & resize the image to 256×256.
  2. Compute a Sobel-magnitude score map as a proxy for "edge-y" pixels.
  3. SLIC into `--n_segments` regions.
  4. Pick top-k regions by mean score and mark them as the "patch of interest".
  5. Save the full label map as `mask.png` (uint16 — IDs survive recursion)
     and a boundary visualization as `superpixels.png`.

Usage:
    python super_pixel_generation.py \\
        --input_image path/to/image.png \\
        --output_dir path/to/figures_dir \\
        [--n_segments 50] [--compactness 10] [--sigma 1.0] [--topk 20]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from skimage import color, filters, io
from skimage.segmentation import slic, find_boundaries
from skimage.transform import resize
from skimage.util import img_as_float


@dataclass
class SuperpixelMaskResult:
    labels: np.ndarray          # (H, W) int — per-pixel superpixel id
    mask: np.ndarray            # (H, W) uint8 — 1 on chosen superpixels
    chosen_ids: np.ndarray      # (K,) chosen superpixel ids


def compute_superpixels_slic(
    img: np.ndarray, n_segments: int = 300, compactness: float = 10.0, sigma: float = 1.0,
) -> np.ndarray:
    """SLIC segmentation. Returns (H, W) int32 labels in {0..K-1}."""
    if img.ndim not in (2, 3):
        raise ValueError("img must be (H,W) or (H,W,3)")
    labels = slic(
        img_as_float(img),
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1 if img.ndim == 3 else None,
    )
    return labels.astype(np.int32)


def pick_topk_by_score(
    labels: np.ndarray, score_map: np.ndarray, topk: int,
) -> np.ndarray:
    """Return the top-k superpixel ids by mean score."""
    if score_map.shape != labels.shape:
        raise ValueError(
            f"score_map shape {score_map.shape} must match labels {labels.shape}"
        )
    K = int(labels.max()) + 1
    flat_lab = labels.reshape(-1)
    flat_score = score_map.reshape(-1).astype(np.float64)

    sums = np.bincount(flat_lab, weights=flat_score, minlength=K)
    counts = np.maximum(np.bincount(flat_lab, minlength=K).astype(np.float64), 1.0)
    sp_scores = sums / counts

    k = int(min(max(topk, 1), K))
    return np.argsort(-sp_scores)[:k].astype(np.int32)


def mask_by_superpixels_topk(
    img: np.ndarray, score_map: np.ndarray, n_segments: int = 300,
    compactness: float = 10.0, sigma: float = 1.0, topk: int = 20,
) -> SuperpixelMaskResult:
    """SLIC + top-k by score. The only mode the pipeline uses."""
    labels = compute_superpixels_slic(
        img, n_segments=n_segments, compactness=compactness, sigma=sigma,
    )
    chosen = pick_topk_by_score(labels, score_map, topk)
    mask = np.isin(labels, chosen).astype(np.uint8)
    return SuperpixelMaskResult(labels=labels, mask=mask, chosen_ids=chosen)


def boundaries_overlay(
    img: np.ndarray, labels: np.ndarray, boundary_value: float | int = 1.0,
) -> np.ndarray:
    """Return a float image with superpixel boundaries highlighted."""
    b = find_boundaries(labels, mode="thick")
    out = img_as_float(img).copy()
    if out.ndim == 2:
        out[b] = boundary_value
    else:
        out[b, :] = boundary_value
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate a SLIC superpixel mask from an image.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--n_segments", type=int, default=50, help="Number of superpixels")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
    parser.add_argument("--topk", type=int, default=20, help="Top-k superpixels to highlight")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write mask.png + superpixels.png")
    args = parser.parse_args()

    img = io.imread(args.input_image)
    print(f"Original shape: {img.shape}")

    if img.shape[0] != 256 or img.shape[1] != 256:
        img = resize(img, (256, 256), anti_aliasing=True)
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    gray = color.rgb2gray(img) if img.ndim == 3 else img.astype(np.float32)
    score = np.abs(filters.sobel(gray))

    res = mask_by_superpixels_topk(
        img, score_map=score,
        n_segments=args.n_segments, compactness=args.compactness, sigma=args.sigma,
        topk=args.topk,
    )
    print(f"Original superpixels: {res.labels.max() + 1}")

    # Save as uint16 so IDs > 255 survive (matches load_superpixel_mask's 16-bit path).
    io.imsave(f"{args.output_dir}/mask.png", res.labels.astype(np.uint16), check_contrast=False)

    vis = boundaries_overlay(
        img, res.labels, boundary_value=255 if img.dtype == np.uint8 else 1.0,
    )
    if vis.dtype != np.uint8:
        vis = (vis * 255).astype(np.uint8)
    io.imsave(f"{args.output_dir}/superpixels.png", vis)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
