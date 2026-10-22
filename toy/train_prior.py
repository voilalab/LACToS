"""Train the diffusion prior of one toy dataset: the DDAD UNet that the toy experiments sample from.

    CUDA_VISIBLE_DEVICES=0 python toy/train_prior.py --dataset quad_L1        # ~8.5 h on one RTX 6000 Ada

Reproduces DDAD's training (baselines/ddad_lib, config toy/ddad_config.yaml): UNetModel(256, 64,
dropout 0, 4 heads, 3 channels), epsilon-prediction loss summed over pixels and averaged over the
batch, Adam lr 3e-4 with weight decay 0.05, batch 32, 1000-step linear beta schedule 1e-4..0.02,
seed 42, on <data_root>/<dataset>/train/good (5000 images in [-1, 1]).

Checkpoints go to <out>/<dataset>/<epoch>. DDAD saves at the FIRST step of every 250th epoch,
so the file named 250 holds the weights after 249 complete epochs; the paper's priors are those
files, and this script keeps the convention (--save_every). One prior per dataset: quad_L1..L4
(toy 1), quadp_p40..p10 (toy 2); quadp_p50 is quad_L1's law and reuses its prior.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines" / "ddad_lib"))
from unet import UNetModel                                   # noqa: E402  (baselines/ddad_lib/unet.py)


class TrainImages(torch.utils.data.Dataset):
    """<folder>/*.png in sorted order, resized to image_size and scaled to [-1, 1] (DDAD's Dataset_maker)."""

    def __init__(self, folder: Path, image_size: int):
        self.paths = sorted(folder.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"no PNGs under {folder}")
        self.tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                      transforms.Lambda(lambda t: t * 2 - 1)])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_unet(cfg) -> torch.nn.Module:
    if cfg.model.DDADS:
        return UNetModel(cfg.data.image_size, 32, dropout=0.3, n_heads=2, in_channels=cfg.data.input_channel)
    return UNetModel(cfg.data.image_size, 64, dropout=0.0, n_heads=4, in_channels=cfg.data.input_channel)


def epsilon_loss(model, x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """DDAD's loss (baselines/ddad_lib/loss.py): ||e - eps_theta(sqrt(a_t) x0 + sqrt(1 - a_t) e, t)||^2."""
    e = torch.randn_like(x0)
    at = alpha_bar.index_select(0, t).view(-1, 1, 1, 1)
    x = at.sqrt() * x0 + (1 - at).sqrt() * e
    return (e - model(x, t.float())).square().sum(dim=(1, 2, 3)).mean(dim=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="quad_L1 .. quad_L4, quadp_p40 .. quadp_p10")
    p.add_argument("--data_root", default=str(ROOT / "toy" / "data"))
    p.add_argument("--out", default=str(ROOT / "toy" / "checkpoints"), help="checkpoints go to <out>/<dataset>/<epoch>")
    p.add_argument("--config", default=str(ROOT / "toy" / "ddad_config.yaml"))
    p.add_argument("--epochs", type=int, default=None, help="default: the config's 250")
    p.add_argument("--save_every", type=int, default=250, help="save at the first step of every N-th epoch (DDAD's convention)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0, help="train on the first N images only (smoke tests)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = OmegaConf.load(a.config)
    epochs = a.epochs or int(cfg.model.epochs)
    batch_size = a.batch_size or int(cfg.data.batch_size)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out) / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(a.seed)
    model = build_unet(cfg).to(device).train()
    print(f"{a.dataset}: UNet with {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters on {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.model.learning_rate),
                                 weight_decay=float(cfg.model.weight_decay))
    data = TrainImages(Path(a.data_root) / a.dataset / "train" / "good", int(cfg.data.image_size))
    if a.max_images:
        data.paths = data.paths[:a.max_images]
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True,
                                         num_workers=int(cfg.model.num_workers), drop_last=True)
    T = int(cfg.model.trajectory_steps)
    betas = np.linspace(float(cfg.model.beta_start), float(cfg.model.beta_end), T, dtype=np.float64)
    alpha_bar = (1 - torch.tensor(betas).float().to(device)).cumprod(dim=0)
    print(f"{len(data)} images, {len(loader)} steps/epoch, {epochs} epochs, batch {batch_size}, "
          f"saving every {a.save_every} epochs to {out_dir}", flush=True)

    t0 = time.time()
    for epoch in range(epochs):
        losses = []
        for step, x0 in enumerate(loader):
            if (epoch + 1) % a.save_every == 0 and step == 0:   # DDAD: at the first step of the epoch
                torch.save(model.state_dict(), out_dir / str(epoch + 1))
                print(f"saved {out_dir / str(epoch + 1)} (after {epoch} complete epochs)", flush=True)
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.shape[0],), device=device).long()
            optimizer.zero_grad()
            loss = epsilon_loss(model, x0, t, alpha_bar)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1}/{epochs}  loss {np.mean(losses):.2f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    torch.save(model.state_dict(), out_dir / "last")
    print(f"done: {out_dir / 'last'} holds the weights after {epochs} epochs")


if __name__ == "__main__":
    main()
