"""Shared I/O helpers: UNet loading, image ↔ tensor, output layout."""
from pathlib import Path
from typing import Dict

import numpy as np
import sys
import torch
from PIL import Image


# ── DDAD UNet loading ────────────────────────────────────────────────
# The DDAD checkpoints used in this repo are DataParallel state dicts with
# a `module.` prefix. We strip it and instantiate the non-parallel UNet.

def _ensure_ddad_on_path() -> None:
    """Prepend the bundled baselines/ddad_lib/ dir to sys.path so DDAD.unet can be imported."""
    root = Path(__file__).resolve().parent.parent.parent
    ddad = str(root / "baselines" / "ddad_lib")
    if ddad not in sys.path:
        sys.path.insert(0, ddad)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first = next(iter(state_dict))
    if first.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_ddad_unet(ckpt_path: str, device: str = "cuda:0") -> torch.nn.Module:
    """Load the DDAD UNet architecture + weights from `ckpt_path`.

    Architecture follows DDAD's `main.py build_model` for `DDADS=False`:
    `UNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)`.

    Raises on strict-load mismatch, which indicates the checkpoint doesn't
    match DDAD's expected architecture.
    """
    _ensure_ddad_on_path()
    from unet import UNetModel                                   # type: ignore

    unet = UNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _strip_module_prefix(state)
    unet.load_state_dict(state, strict=True)
    return unet.to(device).eval()


# ── ADM / guided-diffusion UNet loading (Dhariwal & Nichol) ──────────
# For running DPS on top of the OpenAI ADM checkpoints (e.g. the DPS2022
# FFHQ model `ffhq_10m.pt`). These are `learn_sigma=True`, so the UNet
# outputs 6 channels (epsilon ++ variance); the DPS sampler expects a pure
# epsilon prediction, so we wrap it to return the first 3 channels.

# Location of the vendored guided_diffusion package (the DPS2022 repo).
ADM_REPO_DIR = str(Path(__file__).resolve().parent.parent.parent / "sampling" / "dps")  # vendored guided_diffusion

# Architecture of the DPS2022 FFHQ model `ffhq_10m.pt`. Matches that repo's
# configs/model_config.yaml. `model_path` is intentionally omitted here — we
# load the state dict ourselves (strict) so an architecture mismatch fails loudly
# instead of silently random-initialising (create_model swallows load errors).
_ADM_FFHQ_CONFIG = dict(
    image_size=256, num_channels=128, num_res_blocks=1, channel_mult="",
    learn_sigma=True, class_cond=False, use_checkpoint=False,
    attention_resolutions=16, num_heads=4, num_head_channels=64,
    num_heads_upsample=-1, use_scale_shift_norm=True, dropout=0.0,
    resblock_updown=True, use_fp16=False, use_new_attention_order=False,
)

# The single MVTec model shared by all 15 categories
# (checkpoints/mvtec/guided-diffusion/model400000.pt, 69.8M params). Derived
# from the weights themselves — 18 input blocks, channels 96/96/192/192/384/384,
# attention at the 16x16 and 8x8 levels, and plain `Downsample` convs
# (`input_blocks.3.0.op.*`) rather than ResBlock up/down — then confirmed by a
# strict load. This is the architecture the lost configs/model_config.yaml
# described ("256x256 images, 96 channels" per CLAUDE.md).
_ADM_MVTEC_CONFIG = dict(
    image_size=256, num_channels=96, num_res_blocks=2, channel_mult="",
    learn_sigma=True, class_cond=False, use_checkpoint=False,
    attention_resolutions="16,8", num_heads=4, num_head_channels=64,
    num_heads_upsample=-1, use_scale_shift_norm=True, dropout=0.0,
    resblock_updown=False, use_fp16=False, use_new_attention_order=False,
)

ADM_ARCHS = {"ffhq": _ADM_FFHQ_CONFIG, "mvtec": _ADM_MVTEC_CONFIG}


class _EpsilonUNet(torch.nn.Module):
    """Adapt a `learn_sigma` ADM UNet to the epsilon-only interface the DPS
    sampler expects: forward(x, t) -> epsilon (first 3 of 6 output channels)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)[:, :3]


def _ensure_adm_on_path(repo_dir: str = ADM_REPO_DIR) -> None:
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def load_adm_unet(
    ckpt_path: str,
    device: str = "cuda:0",
    arch=None,   # None -> ffhq; a key of ADM_ARCHS ("ffhq"/"mvtec"); or a dict
    repo_dir: str = ADM_REPO_DIR,
) -> torch.nn.Module:
    """Load an ADM / guided-diffusion UNet (e.g. FFHQ `ffhq_10m.pt`) for DPS.

    Builds the architecture via the vendored `guided_diffusion.create_model`,
    loads `ckpt_path` strictly, and returns an epsilon-only wrapper so it drops
    into the DPS sampler exactly like the DDAD UNet does.
    """
    _ensure_adm_on_path(repo_dir)
    from guided_diffusion.unet import create_model                 # type: ignore

    if arch is None:
        cfg = dict(_ADM_FFHQ_CONFIG)
    elif isinstance(arch, str):
        if arch not in ADM_ARCHS:
            raise ValueError(f"unknown ADM arch {arch!r}; expected one of {sorted(ADM_ARCHS)}")
        cfg = dict(ADM_ARCHS[arch])
    else:
        cfg = dict(arch)
    model = create_model(model_path="", **cfg)                     # arch only; load below
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    return _EpsilonUNet(model).to(device).eval()


# ── Image ↔ tensor conversions ──────────────────────────────────────
# The pipeline operates in [-1, 1] float space, (B, C, H, W) layout.

def image_to_tensor(path: Path, device: str = "cuda:0", size: int = 256) -> torch.Tensor:
    """Load a PNG as a `(1, 3, size, size)` tensor in `[-1, 1]`."""
    img = Image.open(str(path)).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert `(3, H, W)` float tensor in `[-1, 1]` to `(H, W, 3)` uint8 in `[0, 255]`."""
    arr = t.detach().cpu().numpy()
    arr = (arr + 1.0) * 127.5
    return np.clip(arr, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def save_image_tensor(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8(t)).save(str(path))


# ── Output layout matching ood/data.py's loader ─────────────────────

def recon_dir(
    out_root: Path, sample_name: str, test_origin: str, bottom_suffix: str = "4"
) -> Path:
    """Return the directory where reconstructions for one sample should land.

    Matches the layout that `ood/data.py:load_reconstructions` reads:
        {out_root}/{sample_name}/{test_origin}_0_{bottom_suffix}/inpainting/{recon,label,input}/
    """
    return out_root / sample_name / f"{test_origin}_0_{bottom_suffix}" / "inpainting"


def prepare_sample_dirs(
    out_root: Path, sample_name: str, test_origin: str, bottom_suffix: str = "4"
) -> Dict[str, Path]:
    """Create the recon/label/input subdirs for one sample; return their paths."""
    base = recon_dir(out_root, sample_name, test_origin, bottom_suffix)
    dirs = {
        "recon": base / "recon",
        "label": base / "label",
        "input": base / "input",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Effective-σ computation (for the local-Gaussian scorer) ─────────

def effective_sigma_at_t_star(
    t_star: int, trajectory_steps: int = 1000,
    beta_start: float = 0.0001, beta_end: float = 0.02,
) -> float:
    """sqrt(1 - α̅_{t*}) with DDAD's linear β schedule.

    Used by samplers that start from `x_{t*}` (e.g. DDAD native); the resulting
    σ is what the local-Gaussian scorer consumes.
    """
    betas = np.linspace(beta_start, beta_end, trajectory_steps, dtype=np.float64)
    alpha_bar = np.cumprod(1 - betas)
    # t_star is 1-indexed in DDAD's convention; protect against off-by-one.
    idx = max(0, min(t_star - 1, trajectory_steps - 1))
    return float(np.sqrt(1.0 - alpha_bar[idx]))
