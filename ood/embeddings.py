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

from typing import Optional, Tuple

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