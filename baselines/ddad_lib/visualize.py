# import matplotlib.pyplot as plt
# from torchvision.transforms import transforms
# import numpy as np
# import torch
# import os
# from dataset import *


# def visualalize_reconstruction(input, recon, target):
#     plt.figure(figsize=(11,11))
#     plt.subplot(1, 3, 1).axis('off')
#     plt.subplot(1, 3, 2).axis('off')
#     plt.subplot(1, 3, 3).axis('off')

#     plt.subplot(1, 3, 1)
#     plt.imshow(show_tensor_image(input))
#     plt.title('input image')
    

#     plt.subplot(1, 3, 2)
#     plt.imshow(show_tensor_mask(recon))
#     plt.title('recon image')

#     plt.subplot(1, 3, 3)
#     plt.imshow(show_tensor_mask(target))
#     plt.title('target image')


#     k = 0
#     while os.path.exists('results/heatmap{}.png'.format(k)):
#         k += 1
#     plt.savefig('results/heatmap{}.png'.format(k))
#     plt.close()


# # def visualize_reconstructed(input, data,s):
# #     fig, axs = plt.subplots(int(len(data)/5),6)
# #     row = 0
# #     col = 1
# #     axs[0,0].imshow(show_tensor_image(input))
# #     axs[0, 0].get_xaxis().set_visible(False)
# #     axs[0, 0].get_yaxis().set_visible(False)
# #     axs[0,0].set_title('input')
# #     for i, img in enumerate(data):
# #         axs[row, col].imshow(show_tensor_image(img))
# #         axs[row, col].get_xaxis().set_visible(False)
# #         axs[row, col].get_yaxis().set_visible(False)
# #         axs[row, col].set_title(str(i))
# #         col += 1
# #         if col == 6:
# #             row += 1
# #             col = 0
# #     col = 6
# #     row = int(len(data)/5)
# #     remain = col * row - len(data) -1
# #     for j in range(remain):
# #         col -= 1
# #         axs[row-1, col].remove()
# #         axs[row-1, col].get_xaxis().set_visible(False)
# #         axs[row-1, col].get_yaxis().set_visible(False)
        
    
        
# #     plt.subplots_adjust(left=0.1,
# #                     bottom=0.1,
# #                     right=0.9,
# #                     top=0.9,
# #                     wspace=0.4,
# #                     hspace=0.4)
# #     k = 0

# #     while os.path.exists(f'results/reconstructed{k}{s}.png'):
# #         k += 1
# #     plt.savefig(f'results/reconstructed{k}{s}.png')
# #     plt.close()



# def visualize(image, noisy_image, GT, pred_mask, anomaly_map, category) :
#     for idx, img in enumerate(image):
#         plt.figure(figsize=(11,11))
#         plt.subplot(1, 2, 1).axis('off')
#         plt.subplot(1, 2, 2).axis('off')
#         plt.subplot(1, 2, 1)
#         plt.imshow(show_tensor_image(image[idx]))
#         plt.title('clear image')

#         plt.subplot(1, 2, 2)

#         plt.imshow(show_tensor_image(noisy_image[idx]))
#         plt.title('reconstructed image')
#         plt.savefig('results/{}sample{}.png'.format(category,idx))
#         plt.close()

#         plt.figure(figsize=(11,11))
#         plt.subplot(1, 3, 1).axis('off')
#         plt.subplot(1, 3, 2).axis('off')
#         plt.subplot(1, 3, 3).axis('off')

#         plt.subplot(1, 3, 1)
#         plt.imshow(show_tensor_mask(GT[idx]))
#         plt.title('ground truth')

#         plt.subplot(1, 3, 2)
#         plt.imshow(show_tensor_mask(pred_mask[idx]))
#         plt.title('normal' if torch.max(pred_mask[idx]) == 0 else 'abnormal', color="g" if torch.max(pred_mask[idx]) == 0 else "r")

#         plt.subplot(1, 3, 3)
#         plt.imshow(show_tensor_image(anomaly_map[idx]))
#         heatmap_np = anomaly_map[idx].cpu().numpy() # Ensure it's on CPU and is numpy array
#         np.save('results/{}sample{}_heatmap.npy'.format(category, idx), heatmap_np)
#         plt.title('heat map')
#         plt.savefig('results/{}sample{}heatmap.png'.format(category,idx))
#         plt.close()



# def show_tensor_image(image):
#     reverse_transforms = transforms.Compose([
#         transforms.Lambda(lambda t: (t + 1) / (2)),
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#         transforms.Lambda(lambda t: t * 255.),
#         transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
#     ])

#     # Takes the first image of batch
#     if len(image.shape) == 4:
#         image = image[0, :, :, :] 
#     return reverse_transforms(image)

# def show_tensor_mask(image):
#     reverse_transforms = transforms.Compose([
#         # transforms.Lambda(lambda t: (t + 1) / (2)),
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#         transforms.Lambda(lambda t: t.cpu().numpy().astype(np.int8)),
#     ])

#     # Takes the first image of batch
#     if len(image.shape) == 4:
#         image = image[0, :, :, :] 
#     return reverse_transforms(image)
        

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
import os
import PIL.Image as _PIL
from dataset import *


# ── NeurIPS-style inferno overlay (matches evaluate.py) ──────────────────────

def save_inferno_overlay(
    image_tensor,           # (C,H,W) or (1,C,H,W) torch tensor, [-1,1] or [0,1]
    anomaly_map_tensor,     # (C,H,W) or (H,W) torch tensor — raw anomaly scores
    save_dir: str,
    stem: str,              # filename prefix, e.g. "bottle_sample3"
    cmap: str = "inferno",
    alpha: float = 0.55,
    valid_mask: np.ndarray = None,   # optional (H,W) uint8 — restrict normalisation
):
    """
    Saves four files into save_dir:
        {stem}_label_image.png      — clean input image
        {stem}_heatmap_only.png     — inferno-coloured score map
        {stem}_overlay.png          — alpha-blended overlay (no borders)
        {stem}_composite.png        — 3-panel NeurIPS figure (input|score|overlay)
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Convert image tensor → float numpy (H,W,3) in [0,1] ─────────────────
    img = image_tensor
    if img.dim() == 4:
        img = img[0]
    # Handle both [-1,1] and [0,1] ranges
    img_np = img.detach().cpu().float()
    if img_np.min() < -0.01:            # [-1,1] → [0,1]
        img_np = (img_np + 1.0) / 2.0
    img_np = img_np.clamp(0, 1).permute(1, 2, 0).numpy()   # (H,W,3)

    # ── Convert anomaly map → float numpy (H,W) ──────────────────────────────
    heat = anomaly_map_tensor
    if isinstance(heat, torch.Tensor):
        heat = heat.detach().cpu().float()
        if heat.dim() == 4:
            heat = heat[0]
        if heat.dim() == 3:
            heat = heat.mean(0)         # (C,H,W) → (H,W) by averaging channels
        heat = heat.numpy()
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Resize img to match heat if needed ───────────────────────────────────
    if img_np.shape[:2] != heat.shape:
        from skimage.transform import resize
        img_np = resize(img_np, heat.shape, order=1,
                        preserve_range=True, anti_aliasing=True).astype(np.float32)

    # ── Save raw label image ──────────────────────────────────────────────────
    # label_path = os.path.join(save_dir, f"{stem}_label_image.png")
    # _PIL.fromarray((img_np * 255).astype(np.uint8)).save(label_path)

    # ── Percentile normalisation ──────────────────────────────────────────────
    region = heat[valid_mask.astype(bool)] if valid_mask is not None else heat.ravel()
    region = region[np.isfinite(region)]
    lo, hi = (np.percentile(region, [1, 99]) if len(region) > 0
              else (float(heat.min()), float(heat.max())))
    heat_norm = np.clip((heat - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    # ── Heatmap-only PNG ─────────────────────────────────────────────────────
    # heat_only_path = os.path.join(save_dir, f"{stem}_heatmap_only.png")
    # plt.imsave(heat_only_path, heat_norm, cmap=cmap)

    # ── Alpha-blended overlay ─────────────────────────────────────────────────
    cm_obj   = plt.get_cmap(cmap)
    heat_rgb = cm_obj(heat_norm)[..., :3]
    overlay  = np.clip((1 - alpha) * img_np + alpha * heat_rgb, 0.0, 1.0)
    overlay_path = os.path.join(save_dir, f"{stem}_overlay.png")
    _PIL.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)

    # ── 3-panel NeurIPS composite ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), dpi=200, facecolor="white")

    axes[0].imshow(img_np);       axes[0].set_title("Input image",   fontsize=11); axes[0].axis("off")
    im = axes[1].imshow(heat_norm, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Anomaly score", fontsize=11); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay);      axes[2].set_title("Overlay",       fontsize=11); axes[2].axis("off")

    plt.tight_layout(pad=0.5)
    composite_path = os.path.join(save_dir, f"{stem}_composite.png")
    plt.savefig(composite_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # print(f"  [overlay] {label_path}")
    # print(f"  [overlay] {heat_only_path}")
    print(f"  [overlay] {overlay_path}")
    print(f"  [overlay] {composite_path}")


# ── Existing helpers ──────────────────────────────────────────────────────────

def visualalize_reconstruction(input, recon, target):
    plt.figure(figsize=(11,11))
    plt.subplot(1, 3, 1).axis('off')
    plt.subplot(1, 3, 2).axis('off')
    plt.subplot(1, 3, 3).axis('off')

    plt.subplot(1, 3, 1)
    plt.imshow(show_tensor_image(input))
    plt.title('input image')

    plt.subplot(1, 3, 2)
    plt.imshow(show_tensor_mask(recon))
    plt.title('recon image')

    plt.subplot(1, 3, 3)
    plt.imshow(show_tensor_mask(target))
    plt.title('target image')

    k = 0
    while os.path.exists('results/heatmap{}.png'.format(k)):
        k += 1
    plt.savefig('results/heatmap{}.png'.format(k))
    plt.close()


def visualize(image, noisy_image, GT, pred_mask, anomaly_map, category):
    for idx, img in enumerate(image):
        # ── existing clear / reconstructed figure ────────────────────────────
        plt.figure(figsize=(11,11))
        plt.subplot(1, 2, 1).axis('off')
        plt.subplot(1, 2, 2).axis('off')
        plt.subplot(1, 2, 1)
        plt.imshow(show_tensor_image(image[idx]))
        plt.title('clear image')
        plt.subplot(1, 2, 2)
        plt.imshow(show_tensor_image(noisy_image[idx]))
        plt.title('reconstructed image')
        plt.savefig('results/{}sample{}.png'.format(category, idx))
        plt.close()

        # ── existing GT / pred_mask / heatmap figure ─────────────────────────
        plt.figure(figsize=(11,11))
        plt.subplot(1, 3, 1).axis('off')
        plt.subplot(1, 3, 2).axis('off')
        plt.subplot(1, 3, 3).axis('off')

        plt.subplot(1, 3, 1)
        plt.imshow(show_tensor_mask(GT[idx]))
        plt.title('ground truth')

        plt.subplot(1, 3, 2)
        plt.imshow(show_tensor_mask(pred_mask[idx]))
        plt.title('normal'   if torch.max(pred_mask[idx]) == 0 else 'abnormal',
                  color="g"  if torch.max(pred_mask[idx]) == 0 else "r")

        plt.subplot(1, 3, 3)
        plt.imshow(show_tensor_image(anomaly_map[idx]))
        heatmap_np = anomaly_map[idx].cpu().numpy()
        np.save('results/{}sample{}_heatmap.npy'.format(category, idx), heatmap_np)
        plt.title('heat map')
        plt.savefig('results/{}sample{}heatmap.png'.format(category, idx))
        plt.close()

        # ── NEW: inferno overlay matching evaluate.py ─────────────────────────
        save_inferno_overlay(
            image_tensor     = image[idx],
            anomaly_map_tensor = anomaly_map[idx],
            save_dir         = "results",
            stem             = f"{category}sample{idx}",
        )


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


def show_tensor_mask(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.int8)),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)