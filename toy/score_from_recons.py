"""Score a bank of reconstructions with LACToS or DDAD, one (256, 256) float32 map per image.

Both detectors read the SAME reconstructions (sampling/generate_recons.py with `save_float: true`),
so a difference between them is the scoring function, not the sampler.

  --mode single   seed 0 only                                    (DDAD and LACToS)
  --mode mean     pixel mean of the first --k seeds, one image   (DDAD: the "K sample mean" rows)
  --mode set      the first --k seeds as LACToS's population     (LACToS only: "K samples")

DDAD scores the unclamped float32 reconstruction (`recon_float/`), as its own pipeline does;
LACToS scores the 8-bit PNG. LACToS's settings are the toy recipe of the paper: ResNet-101
layer1-3 features projected on a 5-dim PCA basis of the test image (ood.embeddings.embed_and_project),
superpixels of target area 400 px (~169 per image), 32 intensity and 16 feature bins, histogram
sigma 0.1. The reconstructions are fed in filename order (0, 10, 11, .., 19, 1, 2, .., 9 for 20
seeds); cuDNN's numerics depend on the batch composition, so this order is kept fixed.

  CUDA_VISIBLE_DEVICES=0 python toy/score_from_recons.py --method lactos --mode set --k 20 \\
      --recons results/toy/quad/L1/recons/additive_dps_s0.30 --image_dir toy/data/quad_L1/test/q1_square \\
      --figures_root results/toy/quad/L1/figures --out_dir results/toy/quad/L1/additive_dps_s0.30/set20/maps/lactos
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from skimage import io

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODES = ("single", "mean", "set")
TEST_ORIGIN = "ddad_dps_whole"      # directory segment the additive_dps sampler writes
BOTTOM_SUFFIX = "4"
_KINDS = {"png": ("recon", "*.png"), "float": ("recon_float", "*.npy")}

# LACToS toy recipe
SUPERPIXELS = dict(var_threshold=0.0, min_pixels=20, max_sub=6, max_depth=4, compactness=12.0, alpha_grad=10.0)
TARGET_SIZE = 400
EMBEDDINGS = dict(backbone="resnet101", layers=("layer1", "layer2", "layer3"), use_patch_context=True,
                  patchify_size=3, proj_dim_per_layer=None)
SCORING = dict(bins_rgb=32, bins_pca=16, smooth_sigma=0.1, min_pixels=2)
N_PCA = 5


def check_method_mode(method: str, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if method == "ddad" and mode == "set":
        raise ValueError("mode 'set' is undefined for ddad: its map compares ONE reconstruction with the input")


def reference_order(n: int) -> List[int]:
    """Seed order of ood/data.py:load_reconstructions: `<seed>_0_00000.png` sorted as strings."""
    return sorted(range(n), key=lambda s: f"{s}_0_00000.png")


def load_recon_stack(recons_root: Path, sid: str, test_origin: str = TEST_ORIGIN,
                     bottom_suffix: str = BOTTOM_SUFFIX, kind: str = "png",
                     first_k: Optional[int] = None) -> np.ndarray:
    """(N, H, W, 3), stack[k] = seed k. kind: png (uint8) or float (float32 in [-1, 1]).
    first_k: only seeds 0..first_k-1, all of which must exist."""
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(_KINDS)}")
    sub, pattern = _KINDS[kind]
    d = Path(recons_root) / f"samples_{sid}" / f"{test_origin}_0_{bottom_suffix}" / "inpainting" / sub
    seed_of = lambda f: int(Path(f).name.split("_")[0])
    files = sorted(glob(str(d / pattern)), key=seed_of)
    if first_k is not None:
        files = [f for f in files if seed_of(f) < first_k]
        if len(files) != first_k:
            raise FileNotFoundError(f"{d}: need seeds 0..{first_k - 1}, found {len(files)} of them")
    if not files:
        raise FileNotFoundError(f"no {kind} reconstructions under {d}")
    if kind == "float":
        return np.stack([np.load(f).astype(np.float32) for f in files])
    imgs = []
    for f in files:
        img = io.imread(f)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        imgs.append(img)
    return np.asarray(imgs, dtype=np.uint8)


def apply_mode(stack: np.ndarray, mode: str) -> np.ndarray:
    """(N, H, W, 3) -> what the scorer consumes: (1, ...) for single / mean, (N, ...) in filename order for set.
    The mean keeps the stack's representation: rounded uint8 for PNGs, the exact float mean otherwise."""
    if mode == "single":
        return stack[:1]
    if mode == "mean":
        mean = stack.astype(np.float64).mean(axis=0)
        if stack.dtype == np.uint8:
            return np.clip(np.rint(mean), 0, 255).astype(np.uint8)[None]
        return mean.astype(stack.dtype)[None]
    if mode == "set":
        return stack[reference_order(len(stack))]
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def ensure_superpixel_mask(figures_dir: Path, image_path: Path) -> np.ndarray:
    """The SLIC mask that recursive_subdivide refines, computed once per image (super_pixel_generation.py)."""
    from ood.data import load_superpixel_mask
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not (figures_dir / "mask.png").exists():
        subprocess.run([sys.executable, "super_pixel_generation.py", f"--input_image={image_path}",
                        f"--output_dir={figures_dir}"], check=True, cwd=str(ROOT))
    return load_superpixel_mask(str(figures_dir))


# ── DDAD ────────────────────────────────────────────────────────────────────────────────

def load_image(path: Path, size: int = 256) -> torch.Tensor:
    """(1, 3, size, size) in [-1, 1], DDAD's Dataset_maker convention."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1))[None]


def build_ddad(config_path: Path, device: str):
    """DDAD's frozen ImageNet feature extractor and config (baselines/ddad_lib)."""
    sys.path.insert(0, str(ROOT / "baselines" / "ddad_lib"))
    from feature_extractor import domain_adaptation      # type: ignore
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    config.model.device = device
    if not config.model.get("use_frozen_fe", False):
        raise ValueError(f"{config_path}: use_frozen_fe must be true (no UNet is loaded for scoring)")
    fe = domain_adaptation(None, config, fine_tune=False)
    fe.eval()
    return config, fe


def score_ddad(recon: np.ndarray, image_path: Path, config, fe, device: str) -> np.ndarray:
    """heat_map(reconstruction, input): feature distance + v * pixel distance, blurred (sigma 4)."""
    from anomaly_map import heat_map                     # type: ignore
    x = load_image(image_path, config.data.image_size).to(device)
    r = recon.astype(np.float32)
    if recon.dtype == np.uint8:
        r = r / 255.0 * 2.0 - 1.0
    r = torch.from_numpy(r).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        amap = heat_map(r, x, fe, config, use_features=True)
        amap = torch.nn.functional.interpolate(amap.reshape(1, 1, *amap.shape[-2:]).float(),
                                               size=(256, 256), mode="bilinear", align_corners=False)
    return amap[0, 0].cpu().numpy().astype(np.float32)


# ── LACToS ──────────────────────────────────────────────────────────────────────────────

def build_lactos(device: str):
    from ood.embeddings import ResNetPixelEmbedder
    return ResNetPixelEmbedder(resnet_name=EMBEDDINGS["backbone"], layers=tuple(EMBEDDINGS["layers"]),
                               use_patch_context=EMBEDDINGS["use_patch_context"],
                               patchify_size=EMBEDDINGS["patchify_size"],
                               proj_dim_per_layer=EMBEDDINGS["proj_dim_per_layer"]).to(device).eval()


def score_lactos(recon_all: np.ndarray, label_image: np.ndarray, sp_mask: np.ndarray, embedder,
                 target_size: int, device: str) -> np.ndarray:
    """Superpixel refinement -> PCA features -> per-superpixel typicality (NLL - H), unsmoothed."""
    from ood.embeddings import embed_and_project
    from ood.scoring import compute_delta_map_gpu
    from ood.superpixels import recursive_subdivide
    labels_fine, _, _ = recursive_subdivide(img=label_image, labels=sp_mask, **dict(SUPERPIXELS, target_size=target_size))
    emb = embed_and_project(embedder=embedder, label_image=label_image, images_recon_all=recon_all,
                            n_pca=N_PCA, device=device)
    delta_map, _, _ = compute_delta_map_gpu(
        labels_fine=labels_fine, images_recon_all=recon_all, pca_feats_recon=emb["pca_feats_recon"],
        label_image=label_image, label_pca_map=emb["label_pca_map"], device=device,
        gray_scale=False, use_pca=True, **SCORING)
    return np.asarray(delta_map, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=["ddad", "lactos"])
    p.add_argument("--mode", required=True, choices=list(MODES))
    p.add_argument("--k", type=int, default=None, help="seeds 0..k-1 enter the mean / the set (default: every seed present)")
    p.add_argument("--recons", required=True, help="bank: <recons>/samples_<id>/...")
    p.add_argument("--image_dir", required=True, help="the test images <id>.png")
    p.add_argument("--out_dir", required=True, help="one <id>.npy per image")
    p.add_argument("--figures_root", default=None, help="lactos: per-image dirs holding the SLIC mask.png")
    p.add_argument("--ddad_config", default=str(ROOT / "toy" / "ddad_config.yaml"))
    p.add_argument("--ddad_input", default="float", choices=["float", "png"])
    p.add_argument("--target_size", type=int, default=TARGET_SIZE, help="lactos: superpixel target area in px")
    p.add_argument("--sids", nargs="*", default=None, help="subset of image ids (default: every PNG of --image_dir)")
    args = p.parse_args()
    try:
        check_method_mode(args.method, args.mode)
    except ValueError as e:
        p.error(str(e))
    if args.method == "lactos" and not args.figures_root:
        p.error("--figures_root is required for --method lactos")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kind = args.ddad_input if args.method == "ddad" else "png"
    first_k = 1 if args.mode == "single" else args.k
    image_dir, recons_root, out_dir = Path(args.image_dir), Path(args.recons), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sids = args.sids or sorted(q.stem for q in image_dir.glob("*.png"))
    todo = [s for s in sids if not (out_dir / f"{s}.npy").exists()]
    print(f"{args.method}/{args.mode}{'' if first_k is None else first_k} <- {recons_root} ({kind}): "
          f"{len(todo)}/{len(sids)} images to score on {device}", flush=True)
    if not todo:
        return
    if args.method == "ddad":
        config, fe = build_ddad(Path(args.ddad_config), device)
    else:
        from ood.data import load_label_image
        embedder = build_lactos(device)
    for sid in todo:
        stack = load_recon_stack(recons_root, sid, kind=kind, first_k=first_k)
        recon_all = apply_mode(stack, args.mode)
        if args.method == "ddad":
            amap = score_ddad(recon_all[0], image_dir / f"{sid}.png", config, fe, device)
        else:
            label_image = load_label_image(results_dir=str(recons_root), sample_name=f"samples_{sid}",
                                           test_origin=TEST_ORIGIN, bottom_suffix=BOTTOM_SUFFIX)
            sp_mask = ensure_superpixel_mask(Path(args.figures_root) / f"samples_{sid}", image_dir / f"{sid}.png")
            amap = score_lactos(recon_all, label_image, sp_mask, embedder, args.target_size, device)
        np.save(out_dir / f"{sid}.npy", amap)
        print(f"  {sid}: N={recon_all.shape[0]} finite {100 * np.isfinite(amap).mean():.1f}%", flush=True)
    print(f"wrote maps to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
