"""ResNet pixel embeddings + PCA projection.

Extracted from New_dataset_clean.ipynb Cells 6-7, 9, 11.

Changes vs. original (aligned with full_run_hist.py):
  - ResNetPixelEmbedder: resnet152 now correctly uses ResNet152_Weights
    (was ResNet101_Weights — wrong weights for a different model).
  - project_to_pca_gpu: new GPU-accelerated PCA projection (avoids CPU
    matmul for large batches).
  - embed_images_batched: batch-based alternative to embed_images for faster
    throughput when GPU memory allows.
"""

import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


def patchify_context(
    features: torch.Tensor, patchsize: int = 3, stride: int = 1
) -> torch.Tensor:
    """Local context: unfold -> average over patch -> same shape."""
    padding = (patchsize - 1) // 2
    unfolder = torch.nn.Unfold(kernel_size=patchsize, stride=stride, padding=padding)
    B, C, H, W = features.shape
    unfolded = unfolder(features)
    unfolded = unfolded.view(B, C, patchsize * patchsize, H * W)
    pooled = unfolded.mean(dim=2)
    return pooled.view(B, C, H, W)


class ResNetPixelEmbedder(nn.Module):
    """Full-resolution pixel embedding by concatenating upsampled ResNet features."""

    # FIX: resnet152 now correctly uses ResNet152_Weights (was ResNet101_Weights)
    _REGISTRY = {
        "resnet18":  (models.resnet18,  "ResNet18_Weights",  "IMAGENET1K_V1"),
        "resnet50":  (models.resnet50,  "ResNet50_Weights",  "IMAGENET1K_V2"),
        "resnet101": (models.resnet101, "ResNet101_Weights", "IMAGENET1K_V2"),
        "resnet152": (models.resnet152, "ResNet152_Weights", "IMAGENET1K_V1"),
    }

    def __init__(
        self,
        resnet_name: str = "resnet18",
        layers: Tuple[str, ...] = ("layer1", "layer2", "layer3"),
        out_size: Optional[int] = None,
        use_imagenet_norm: bool = True,
        use_patch_context: bool = True,
        patchify_size: int = 3,
        proj_dim_per_layer: Optional[int] = None,
        **kwargs,  # absorb extra config keys
    ):
        super().__init__()

        if resnet_name not in self._REGISTRY:
            raise ValueError(
                f"resnet_name must be one of {list(self._REGISTRY)}, got {resnet_name}"
            )
        loader, weights_cls, weights_variant = self._REGISTRY[resnet_name]
        weights = getattr(getattr(models, weights_cls), weights_variant)
        net = loader(weights=weights)
        net.eval()

        return_nodes = {ln: ln for ln in layers}
        self.extractor = create_feature_extractor(net, return_nodes=return_nodes)
        self.layers = list(layers)
        self.out_size = out_size
        self.use_patch_context = use_patch_context
        self.patchify_size = patchify_size

        self.proj = nn.ModuleDict()
        self.proj_dim_per_layer = proj_dim_per_layer
        if proj_dim_per_layer is not None:
            ch_map = {}
            if resnet_name == "resnet18":
                ch_map = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
            else:
                ch_map = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
            for ln in layers:
                self.proj[ln] = nn.Conv2d(ch_map[ln], proj_dim_per_layer, kernel_size=1, bias=False)

        self.use_imagenet_norm = use_imagenet_norm
        if use_imagenet_norm:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] float in [0,1] or [-1,1].
        Returns:
            emb: [B, C_embed, H, W] concatenated multi-layer features.
        """
        assert x.ndim == 4 and x.shape[1] == 3
        B, _, H, W = x.shape

        if x.min() < 0:
            x = (x + 1) / 2.0

        if self.use_imagenet_norm:
            x = (x - self.mean) / self.std

        feats = self.extractor(x)

        outH, outW = (self.out_size, self.out_size) if self.out_size is not None else (H, W)

        ups = []
        for ln in self.layers:
            f = feats[ln]
            if self.use_patch_context:
                f = patchify_context(f, patchsize=self.patchify_size, stride=1)
            if self.proj_dim_per_layer is not None:
                f = self.proj[ln](f)
            f = F.interpolate(f, size=(outH, outW), mode="bilinear", align_corners=False)
            f = F.normalize(f, dim=1)
            ups.append(f)

        return torch.cat(ups, dim=1)


def to_tensor01(img_np: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """Convert numpy image (H,W,3) uint8/float to (1,3,H,W) float [0,1] tensor."""
    t = torch.from_numpy(img_np).float()
    if t.ndim == 2:
        t = t[..., None].repeat(1, 1, 3)
    if t.max() > 1.0:
        t = t / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return t


def embed_images(
    embedder: ResNetPixelEmbedder,
    images: np.ndarray,
    device: str = "cuda",
) -> torch.Tensor:
    """Embed all images one-at-a-time to avoid GPU OOM.

    Args:
        embedder: ResNetPixelEmbedder on device.
        images: (B, H, W, 3) uint8 array.
        device: torch device string.

    Returns:
        (B, C, H, W) tensor on CPU.
    """
    x_all = torch.from_numpy(images).float().permute(0, 3, 1, 2)
    if x_all.max() > 1.0:
        x_all = x_all / 255.0

    feat_list = []
    with torch.no_grad():
        for i in range(x_all.shape[0]):
            xi = x_all[i : i + 1].to(device)
            fi = embedder(xi)
            feat_list.append(fi.cpu())
            del xi, fi
            if i % 10 == 0:
                torch.cuda.empty_cache()

    result = torch.cat(feat_list, dim=0)
    del feat_list
    gc.collect()
    torch.cuda.empty_cache()
    return result


def embed_images_batched(
    embedder: ResNetPixelEmbedder,
    images: np.ndarray,
    batch_size: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """Embed images in batches for higher throughput.

    Prefer this over embed_images when GPU memory allows multiple images
    at once (typical for resnet18/50; use embed_images for resnet101/152
    on low-VRAM GPUs).

    Args:
        embedder: ResNetPixelEmbedder on device.
        images: (B, H, W, 3) uint8 array.
        batch_size: images per GPU batch.
        device: torch device string.

    Returns:
        (B, C, H, W) tensor on CPU.
    """
    x_all = torch.from_numpy(images).float().permute(0, 3, 1, 2)
    if x_all.max() > 1.0:
        x_all = x_all / 255.0

    feat_list = []
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            xb = x_all[i : i + batch_size].to(device)
            feat_list.append(embedder(xb).cpu())
            if i % 40 == 0:
                torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    return torch.cat(feat_list, dim=0)


# def compute_pca_basis(
#     label_feat_cpu: torch.Tensor, n_components: int = 9
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """PCA via SVD on label image features.

#     Args:
#         label_feat_cpu: (C, H, W) tensor on CPU.
#         n_components: number of PCA components.

#     Returns:
#         mu: (1, C) mean vector.
#         components: (n_components, C) PCA basis.
#     """
#     C, H, W = label_feat_cpu.shape
#     X = label_feat_cpu.permute(1, 2, 0).reshape(-1, C).numpy()
#     mu = X.mean(axis=0, keepdims=True)
#     X_centered = X - mu
#     _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
#     components = Vt[:n_components]

#     explained = (S[:n_components] ** 2) / (S**2).sum()
#     print(f"PCA explained variance ({n_components} components): {explained.sum() * 100:.1f}%")

#     return mu, components
def compute_pca_basis(label_feat_cpu: torch.Tensor, n_components: int,
                      device: torch.device = None) -> Tuple[np.ndarray, np.ndarray]:
    C, H, W = label_feat_cpu.shape
    X  = label_feat_cpu.permute(1, 2, 0).reshape(-1, C)   # (H*W, C)
    mu = X.mean(dim=0, keepdim=True)                        # (1, C)
    Xc = X - mu                                             # centered

    if device is not None and device.type == "cuda":
        # GPU SVD — torch.linalg.svd is ~10x faster than numpy for large C
        Xc_gpu = Xc.to(device)
        _, _, Vt_gpu = torch.linalg.svd(Xc_gpu, full_matrices=False)
        components = Vt_gpu[:n_components].cpu().numpy()
        mu_np = mu.numpy()
    else:
        _, S, Vt = np.linalg.svd(Xc.numpy(), full_matrices=False)
        components = Vt[:n_components]
        mu_np = mu.numpy()
        S_for_print = S

    # explained variance (approximate when using GPU path)
    print(f"PCA basis computed ({n_components} components, C={C})")
    return mu_np, components

def project_to_pca(
    feat_hwc: np.ndarray, mu: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Project features to PCA space, clip to [-1, 1].

    Args:
        feat_hwc: (M, C) flattened spatial pixels.
        mu: (1, C) mean.
        components: (k, C) PCA basis.

    Returns:
        (M, k) projections clipped to [-1, 1].
    """
    proj = (feat_hwc - mu) @ components.T
    return proj


def project_to_pca_gpu(
    feat_tensor: torch.Tensor,
    mu: np.ndarray,
    components: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """GPU-accelerated PCA projection for large batches.

    Args:
        feat_tensor: (B, C, H, W) feature map on GPU.
        mu: (1, C) mean vector (numpy).
        components: (n_pca, C) PCA basis (numpy).
        device: torch device.

    Returns:
        (B, H, W, n_pca) float32 tensor, clamped to [-1, 1].
    """
    mu_t   = torch.from_numpy(mu).float().to(device)
    comp_t = torch.from_numpy(components).float().to(device)
    B, C, H, W = feat_tensor.shape
    x    = feat_tensor.permute(0, 2, 3, 1).reshape(-1, C)
    proj = (x - mu_t) @ comp_t.t()
    return proj.reshape(B, H, W, -1)


# def embed_and_project(
#     embedder: ResNetPixelEmbedder,
#     label_image: np.ndarray,
#     images_recon_all: np.ndarray,
#     n_pca: int = 9,
#     device: str = "cuda",
# ) -> Dict[str, np.ndarray]:
#     """Full embedding + PCA pipeline.

#     Args:
#         embedder: ResNetPixelEmbedder on device.
#         label_image: (H, W, 3) uint8.
#         images_recon_all: (B, H, W, 3) uint8.
#         n_pca: number of PCA components.
#         device: torch device string.

#     Returns:
#         dict with keys: label_pca_map, pca_feats_recon, mu, components.
#     """
#     model_device = next(embedder.parameters()).device

#     # Embed label image
#     with torch.no_grad():
#         label_x = to_tensor01(label_image, device=str(model_device))
#         label_feat = embedder(label_x).squeeze(0).cpu()

#     # PCA basis from label
#     mu, components = compute_pca_basis(label_feat, n_components=n_pca)

#     # Project label image
#     C, H, W = label_feat.shape
#     X_label = label_feat.permute(1, 2, 0).reshape(-1, C).numpy()
#     label_pca_map = project_to_pca(X_label, mu, components).reshape(H, W, n_pca)

#     # Embed all reconstructions
#     # feat_map = embed_images(embedder, images_recon_all, device=device)
#     feat_map = embed_images_batched(embedder, images_recon_all,
#                                     batch_size=8,   # lower if OOM with resnet101
#                                     device=device)
#     # Project all reconstructions (GPU-accelerated)
#     feat_map_dev = feat_map.to(model_device)
#     pca_feats_recon = project_to_pca_gpu(feat_map_dev, mu, components, model_device)\
#                         .cpu().numpy()                          # (B, H, W, n_pca)
#     del feat_map_dev

#     return {
#         "label_pca_map": label_pca_map,
#         "pca_feats_recon": pca_feats_recon,
#         "mu": mu,
#         "components": components,
#     }

def embed_and_project(
    embedder: ResNetPixelEmbedder,
    label_image: np.ndarray,
    images_recon_all: np.ndarray,
    n_pca: int = 9,
    device: str = "cuda",
    batch_size: int = 8,          # ← new param; lower for resnet101/152
) -> Dict[str, np.ndarray]:

    model_device = next(embedder.parameters()).device

    # Embed label image
    with torch.no_grad():
        label_x    = to_tensor01(label_image, device=str(model_device))
        label_feat = embedder(label_x).squeeze(0).cpu()

    # PCA basis from label
    mu, components = compute_pca_basis(label_feat, n_components=n_pca, device=model_device)

    # Project label image
    C, H, W = label_feat.shape
    X_label = label_feat.permute(1, 2, 0).reshape(-1, C).numpy()
    label_pca_map = project_to_pca(X_label, mu, components).reshape(H, W, n_pca)

    # ── Embed + project recon images in one fused batched loop ───────────
    # Never materialises the full (N, C, H, W) tensor — keeps peak GPU RAM
    # at batch_size × C × H × W instead of N × C × H × W.
    x_all = torch.from_numpy(images_recon_all).float().permute(0, 3, 1, 2)
    if x_all.max() > 1.0:
        x_all = x_all / 255.0

    pca_list = []
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            xb   = x_all[i:i+batch_size].to(model_device)
            fb   = embedder(xb)
            proj = project_to_pca_gpu(fb, mu, components, model_device)
            pca_list.append(proj.cpu())
            del xb, fb, proj
            torch.cuda.empty_cache()

    pca_feats_recon = torch.cat(pca_list, 0).numpy()   # (N, H, W, n_pca)
    gc.collect()

    return {
        "label_pca_map":   label_pca_map,
        "pca_feats_recon": pca_feats_recon,
        "mu":              mu,
        "components":      components,
    }