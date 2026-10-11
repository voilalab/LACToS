"""SLIC superpixel mask for one 256x256 image (called by evaluate.py when figures/<sample>/mask.png is missing).

    python super_pixel_generation.py --input_image IMG.png --output_dir DIR   [--n_segments 50 --compactness 10 --sigma 1]
"""
from __future__ import annotations

import argparse

import numpy as np
from skimage import io
from skimage.segmentation import slic
from skimage.transform import resize
from skimage.util import img_as_float



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


def main():
    parser = argparse.ArgumentParser(description="Generate the SLIC superpixel mask of one image (mask.png, uint16 ids).")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--n_segments", type=int, default=50)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian pre-smoothing of SLIC")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    img = io.imread(args.input_image)
    if img.shape[0] != 256 or img.shape[1] != 256:
        img = resize(img, (256, 256), anti_aliasing=True)
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    labels = compute_superpixels_slic(img, n_segments=args.n_segments, compactness=args.compactness, sigma=args.sigma)
    # uint16 so that ids > 255 survive (ood.data.load_superpixel_mask reads 16-bit)
    io.imsave(f"{args.output_dir}/mask.png", labels.astype(np.uint16), check_contrast=False)
    print(f"{args.input_image}: {labels.max() + 1} superpixels -> {args.output_dir}/mask.png")


if __name__ == "__main__":
    main()
