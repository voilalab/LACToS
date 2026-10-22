#!/usr/bin/env python
"""Render the score maps that evaluate.py dumps: image | LACToS heatmap (smoothed) | ground truth.

  python scripts/plot_maps.py results/quickstart --sigma 5            # one PNG per samples_<id>.npz, next to it
  python scripts/plot_maps.py results/lactos_maps/CT/sp100_m5_hs0.5_b16x64/cta --sigma 2 --out results/figures/ct --limit 10

The heatmap is the delta map smoothed with a spatial Gaussian of --sigma (the paper: CHAOS 2, FFHQ 5,
DVXray 5), min-max normalised per image inside the valid region and drawn over the image; the
ground-truth mask is outlined in red on the heatmap.
"""
import argparse, glob, os
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(npz_path, sigma, out_path):
    z = np.load(npz_path)
    d = np.nan_to_num(z["delta_map"].astype(np.float32))
    img = z["label_image"]
    gt = z["gt_mask"] > 0
    valid = z["valid_mask"].astype(bool) if z["valid_mask"].shape == d.shape else np.ones(d.shape, bool)
    heat = gaussian_filter(d, sigma, mode="nearest") if sigma > 0 else d
    lo, hi = np.percentile(heat[valid], [1, 99.5])
    heat = np.clip((heat - lo) / max(hi - lo, 1e-8), 0, 1) * valid
    edge = binary_dilation(gt, iterations=2) & ~gt
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(img, cmap="gray" if img.ndim == 2 else None); ax[0].set_title("image")
    ax[1].imshow(img, cmap="gray" if img.ndim == 2 else None); ax[1].imshow(heat, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    ax[1].contour(gt.astype(float), levels=[0.5], colors="red", linewidths=1); ax[1].set_title(f"LACToS score (sigma {sigma:g})")
    ax[2].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[2].set_title("ground truth")
    for a in ax: a.axis("off")
    fig.suptitle(os.path.basename(npz_path)[:-4]); fig.tight_layout()
    fig.savefig(out_path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps_dir", help="directory with samples_<id>.npz dumps (evaluate.py --dump_maps)")
    ap.add_argument("--sigma", type=float, default=5.0, help="spatial Gaussian sigma in pixels")
    ap.add_argument("--out", default=None, help="output directory (default: maps_dir)")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N maps")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.maps_dir, "*.npz")))
    if a.limit: files = files[:a.limit]
    if not files: raise SystemExit(f"no npz maps in {a.maps_dir}")
    out = a.out or a.maps_dir; os.makedirs(out, exist_ok=True)
    for f in files:
        p = os.path.join(out, os.path.basename(f)[:-4] + ".png"); render(f, a.sigma, p); print(p)


if __name__ == "__main__":
    main()
