"""Data loading: DPS reconstructions, label images, masks.

Extracted from New_dataset_clean.ipynb Cells 0-4.
"""

# import os
# from glob import glob
# from typing import Optional

# import numpy as np
# from skimage import io
# import PIL.Image as Image


# def load_reconstructions(
#     results_dir: str,
#     sample_name: str,
#     test_origin: str,
#     num_patches: int,
#     bottom_suffix: str,
# ) -> np.ndarray:
#     """Load all DPS reconstruction images across patches.

#     Scans ``{results_dir}/{sample_name}/{test_origin}_{patch}_{bottom_suffix}/inpainting/recon/*.png``
#     for each patch index 0..num_patches-1, stacks into a single array.

#     Returns:
#         (B, H, W, 3) uint8 array where B = total images across all patches.
#     """
#     all_images = []

#     for patch_idx in range(num_patches):
#         directory = os.path.join(
#             results_dir, sample_name,
#             f"{test_origin}_{patch_idx}_{bottom_suffix}",
#             "inpainting", "recon",
#         )
#         image_paths = sorted(glob(os.path.join(directory, "*.png")))

#         for path in image_paths:
#             img = io.imread(path)
#             if img.ndim == 3 and img.shape[-1] == 4:
#                 img = img[..., :3]
#             all_images.append(img)

#         if not image_paths:
#             print(f"Warning: no images in {directory}")

#     if not all_images:
#         raise FileNotFoundError(
#             f"No reconstruction images found in {results_dir}/{sample_name}/"
#         )

#     # Verify consistent shape
#     shapes = {img.shape for img in all_images}
#     if len(shapes) != 1:
#         raise ValueError(f"Inconsistent image shapes: {shapes}")

#     return np.array(all_images)


# def load_label_image(
#     results_dir: str,
#     sample_name: str,
#     test_origin: str,
#     bottom_suffix: str,
# ) -> np.ndarray:
#     """Load the label (original) image from DPS output.

#     Searches patch indices 0..49 for the first available label file. Label
#     files may be named `0_00000.png` (batched) or `00000.png` (single).

#     Returns:
#         (H, W, 3) uint8 array.
#     """
#     label_names = ["0_00000.png", "00000.png"]
#     for idx in range(50):
#         for lname in label_names:
#             candidate = os.path.join(
#                 results_dir, sample_name,
#                 f"{test_origin}_{idx}_{bottom_suffix}",
#                 "inpainting", "label", lname,
#             )
#             if os.path.exists(candidate):
#                 img = io.imread(candidate)
#                 if img.ndim == 3 and img.shape[-1] == 4:
#                     img = img[..., :3]
#                 return img[:, :, :3]
#     raise FileNotFoundError(
#         f"No label image found in {results_dir}/{sample_name}/ for any patch index"
#     )


# def load_superpixel_mask(figures_dir: str) -> np.ndarray:
#     """Load the superpixel label mask from figures/mask.png.

#     Returns:
#         (H, W) int array of superpixel IDs.
#     """
#     path = os.path.join(figures_dir, "mask.png")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Superpixel mask not found: {path}")
#     mask = np.array(Image.open(path))
#     if mask.ndim == 3:
#         mask = mask[:, :, 0]
#     # Use 16-bit read if available, to support superpixel IDs > 255
#     if mask.dtype == np.uint8 and mask.max() == 255:
#         mask_16 = np.array(Image.open(path).convert("I"))
#         if mask_16.max() > 255:
#             mask = mask_16
#     return mask.astype(np.int32)


# def load_gt_mask(
#     path_template: str,
#     sample: str,
#     downsample_factor: int = 4,
# ) -> np.ndarray:
#     """Load ground-truth anomaly mask and binarize.

#     Args:
#         path_template: path with ``{sample}`` placeholder.
#         sample: sample identifier to substitute.
#         downsample_factor: spatial downsampling factor.

#     Returns:
#         (H, W) uint8 binary mask (1 = anomaly).
#     """
#     path = path_template.format(sample=sample)
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"GT mask not found: {path}")

#     mask = np.array(Image.open(path).convert("L"))
#     if downsample_factor > 1:
#         mask = mask[::downsample_factor, ::downsample_factor]

#     return (mask > 0).astype(np.uint8)


# def parse_sample_id(sample_name: str) -> str:
#     """Extract sample ID from sample_name like 'samples_010' -> '010'."""
#     parts = sample_name.split("_")[1:]
#     return "_".join(parts)


"""Data loading: DPS reconstructions, label images, masks.

Extracted from New_dataset_clean.ipynb Cells 0-4.

Changes vs. original (aligned with full_run_hist.py):
  - load_mask: attempts 16-bit read when 8-bit max==255 to support SP IDs > 255.
  - load_gt_mask: expanded to handle cable / CT / faces categories with
    actual directory layouts (replaces the simple path_template approach).
"""

import os
from glob import glob
from typing import Optional

import numpy as np
from skimage import io
from skimage.transform import resize
import PIL.Image as Image


def load_reconstructions(
    results_dir: str,
    sample_name: str,
    test_origin: str,
    num_patches: int,
    bottom_suffix: str,
) -> np.ndarray:
    """Load all DPS reconstruction images across patches.

    Scans ``{results_dir}/{sample_name}/{test_origin}_{patch}_{bottom_suffix}/inpainting/recon/*.png``
    for each patch index 0..num_patches-1, stacks into a single array.

    Returns:
        (B, H, W, 3) uint8 array where B = total images across all patches.
    """
    all_images = []

    for patch_idx in range(num_patches):
        directory = os.path.join(
            results_dir, sample_name,
            f"{test_origin}_{patch_idx}_{bottom_suffix}",
            "inpainting", "recon",
        )
        image_paths = sorted(glob(os.path.join(directory, "*.png")))

        for path in image_paths:
            img = io.imread(path)
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            all_images.append(img)

        if not image_paths:
            print(f"Warning: no images in {directory}")

    if not all_images:
        raise FileNotFoundError(
            f"No reconstruction images found in {results_dir}/{sample_name}/"
        )

    shapes = {img.shape for img in all_images}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent image shapes: {shapes}")

    return np.array(all_images)


def load_label_image(
    results_dir: str,
    sample_name: str,
    test_origin: str,
    bottom_suffix: str,
) -> np.ndarray:
    """Load the label (original) image from DPS output.

    Searches patch indices 0..49 for the first available label file. Label
    files may be named `0_00000.png` (batched) or `00000.png` (single).

    Returns:
        (H, W, 3) uint8 array.
    """
    label_names = ["0_00000.png", "00000.png"]
    for idx in range(50):
        for lname in label_names:
            candidate = os.path.join(
                results_dir, sample_name,
                f"{test_origin}_{idx}_{bottom_suffix}",
                "inpainting", "label", lname,
            )
            if os.path.exists(candidate):
                img = io.imread(candidate)
                if img.ndim == 3 and img.shape[-1] == 4:
                    img = img[..., :3]
                return img[:, :, :3]
    raise FileNotFoundError(
        f"No label image found in {results_dir}/{sample_name}/ for any patch index"
    )


def load_superpixel_mask(figures_dir: str) -> np.ndarray:
    """Load the superpixel label mask from figures/mask.png.

    Attempts a 16-bit read when the 8-bit image is saturated at 255, to
    support superpixel IDs > 255.

    Returns:
        (H, W) int array of superpixel IDs.
    """
    path = os.path.join(figures_dir, "mask.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Superpixel mask not found: {path}")
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # Upgrade to 16-bit when 8-bit is saturated — may carry IDs > 255
    if mask.dtype == np.uint8 and mask.max() == 255:
        mask_16 = np.array(Image.open(path).convert("I"))
        if mask_16.max() > 255:
            mask = mask_16
    return mask.astype(np.int32)

def load_gt_mask(
    path_template: str = None,
    sample: str = None,
    downsample_factor: int = 4,
    # New-style keyword arguments (used by full_run_hist.py)
    gt_root: str = None,
    defect_type: str = "combined",
    sample_str: str = None,
    target_shape: tuple = None,
    downsample: int = None,
    category: str = "cable",
) -> np.ndarray:
    """Load ground-truth anomaly mask and binarize.

    Supports two calling styles:

    Old style (evaluate.py):
        load_gt_mask(path_template="/path/{sample}_mask.png", sample="010")

    New style (full_run_hist.py):
        load_gt_mask(gt_root=..., defect_type=..., sample_str=...,
                     target_shape=..., downsample=4, category="cable")
    """
    def _fit(m: np.ndarray, shape: tuple) -> np.ndarray:
        if shape is None or m.shape == shape:
            return m
        return resize(
            m, shape, order=0, mode="edge",
            preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)

    # ── Old-style: path_template + sample ────────────────────────────────
    if path_template is not None:
        path = path_template.format(sample=sample)
        if not os.path.exists(path):
            raise FileNotFoundError(f"GT mask not found: {path}")
        mask = np.array(Image.open(path).convert("L"))
        if downsample_factor > 1:
            mask = mask[::downsample_factor, ::downsample_factor]
        return (mask > 0).astype(np.uint8)

    # ── New-style: gt_root + category-aware logic ─────────────────────────
    ds = downsample if downsample is not None else downsample_factor

    if category == "CT":
        if len(sample_str) > 4:
            sample_str = str(int(sample_str))
        gt_path = (
            f"/data2/akheirandish3/id_new_warped_images/"
            f"masks_ood_default_large/{sample_str}_mask.png"
        )
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"CT GT mask not found: {gt_path}")
        mask = np.array(Image.open(gt_path).convert("L"))[::2, ::2]
        return _fit((mask > 0).astype(np.uint8), target_shape)

    if len(sample_str) > 4:
        sample_str = str(int(sample_str))
    for cand in [
        os.path.join(gt_root, f"{sample_str}_mask.png"),
        os.path.join(gt_root, defect_type, f"{sample_str}_mask.png"),
    ]:
        if os.path.exists(cand):
            mask = np.array(Image.open(cand).convert("L"))
            if ds > 1:
                mask = mask[::ds, ::ds]
            return _fit((mask > 0).astype(np.uint8), target_shape)

    raise FileNotFoundError(f"GT mask not found for sample {sample_str} in {gt_root}")


def parse_sample_id(sample_name: str) -> str:
    """Extract sample ID from sample_name like 'samples_010' -> '010'."""
    parts = sample_name.split("_")[1:]
    return "_".join(parts)
