"""Fit PatchCore on a toy dataset's train/good and write one (256, 256) float32 map per test image.

Runs against the upstream clone third_party/patchcore-inspection (scripts/fetch_third_party.sh)
in the `patchcore` conda environment (environment/patchcore.txt).

Recipe is PatchCore's published detection baseline: WideResNet-50, layer2 +
layer3, 10% approximate greedy coreset, 1024 -> 1024 embeddings, patchsize 3,
1 nearest neighbour, resize 256 / imagesize 256 (no center crop) -- this is a
deliberate deviation from the published recipe's imagesize 224. This
benchmark's images are natively 256x256 and its synthetic anomalies (ellipse
centers drawn Uniform over [30, 226] on a 256 canvas) routinely fall in the
outer ~16px border that a 224 center crop would discard, and the naive
224->256 upsample used to satisfy the shared map contract misaligns crop-local
pixel i with original pixel i+16 vs. the resize's i*256/224 -- a 16px radial
shift at the border, larger than the ellipse's 12px semi-minor axis. Running
at imagesize 256 with no crop keeps PatchCore able to see the same full image
the other two methods in this benchmark see, and keeps the saved map
pixel-aligned with the input.

    CUDA_VISIBLE_DEVICES=0 python toy/run_patchcore.py --data_root toy/data/quad_L1 --test_subdirs q1_square good \\
        --n_test 50 --out_root results/toy/quad/L1/patchcore      # -> <out_root>/<subdir>/maps/patchcore/<id>.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
PATCHCORE_REPO = REPO_ROOT / "third_party" / "patchcore-inspection"
if not (PATCHCORE_REPO / "src").is_dir():
    raise SystemExit(f"{PATCHCORE_REPO} missing: run bash scripts/fetch_third_party.sh")
sys.path.insert(0, str(PATCHCORE_REPO / "src"))

import patchcore.backbones          # noqa: E402
import patchcore.common             # noqa: E402
import patchcore.patchcore          # noqa: E402
import patchcore.sampler            # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TorchNN:
    """Exact nearest-neighbour search on one GPU: squared L2, the same quantity faiss's
    IndexFlatL2 returns, so PatchCore's scores are unchanged. faiss here is CPU-only,
    and searching a full (no-coreset) memory bank of ~5M x 1024 on CPU takes hours;
    as chunked matrix products on a GPU it takes seconds."""

    def __init__(self, device, chunk: int = 131072) -> None:
        self.device, self.chunk, self.bank = torch.device(device), int(chunk), None
        torch.backends.cuda.matmul.allow_tf32 = False          # keep full fp32 distances

    def fit(self, features: np.ndarray) -> None:
        self.bank = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(self.device)
        self.bank_sq = (self.bank * self.bank).sum(1)

    @torch.no_grad()
    def run(self, n_nearest_neighbours, query_features, index_features=None):
        if index_features is not None:
            raise NotImplementedError("TorchNN searches the fitted bank only")
        n = int(n_nearest_neighbours)
        q = torch.from_numpy(np.ascontiguousarray(query_features, dtype=np.float32)).to(self.device)
        q_sq = (q * q).sum(1, keepdim=True)
        best_d = torch.full((q.shape[0], n), float("inf"), device=self.device)
        best_i = torch.full((q.shape[0], n), -1, dtype=torch.long, device=self.device)
        for s in range(0, self.bank.shape[0], self.chunk):
            b = self.bank[s:s + self.chunk]
            d = (q_sq + self.bank_sq[s:s + self.chunk][None] - 2.0 * (q @ b.T)).clamp_min_(0.0)
            dv, di = torch.topk(d, min(n, d.shape[1]), dim=1, largest=False)
            cd, ci = torch.cat([best_d, dv], 1), torch.cat([best_i, di + s], 1)
            best_d, pick = torch.topk(cd, n, dim=1, largest=False)
            best_i = torch.gather(ci, 1, pick)
        return best_d.cpu().numpy(), best_i.cpu().numpy()

    def reset_index(self) -> None:
        self.bank = None


class FolderDataset(torch.utils.data.Dataset):
    """PNG folder -> normalised (3, imagesize, imagesize) tensors, in sorted-stem order."""

    def __init__(self, folder: Path, resize: int = 256, imagesize: int = 256, limit: int = 0):
        self.paths = sorted(folder.glob("*.png"), key=lambda p: p.stem)
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no PNGs under {folder}")
        self.resize, self.imagesize = resize, imagesize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = Image.open(self.paths[i]).convert("RGB").resize(
            (self.resize, self.resize), Image.BILINEAR)
        if self.imagesize != self.resize:
            # centre crop, PatchCore's transform -- a no-op when imagesize == resize
            off = (self.resize - self.imagesize) // 2
            img = img.crop((off, off, off + self.imagesize, off + self.imagesize))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1))

    @property
    def ids(self):
        return [p.stem for p in self.paths]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="dataset dir holding train/good and test/<subdir>")
    p.add_argument("--test_subdirs", nargs="+", default=["q1_square"],
                   help="test subfolders to score; the memory bank is fit once (on train/good) and reused")
    p.add_argument("--out_root", required=True, help="maps go to <out_root>/<subdir>/maps/patchcore/<id>.npy")
    p.add_argument("--n_test", type=int, default=0, help="score only the first N images (sorted) of every subdir")
    p.add_argument("--max_train", type=int, default=0, help="fit on the first N training images only (smoke tests)")
    p.add_argument("--imagesize", type=int, default=256)
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--target_size", type=int, default=256, help="Size of the saved map.")
    p.add_argument("--coreset_percentage", type=float, default=0.1,
                   help="fraction of training patches kept in the memory bank; 1.0 keeps all of "
                        "them (IdentitySampler -- the coreset class rejects 1.0 itself)")
    p.add_argument("--nn", choices=["faiss", "torch"], default="faiss",
                   help="nearest-neighbour backend: faiss (CPU, the reported recipe) or an exact "
                        "GPU search returning the same squared-L2 distances")
    p.add_argument("--faiss_workers", type=int, default=4,
                   help="CPU threads for the nearest-neighbour search (faiss has no GPU build here); "
                        "a full memory bank needs many more than the default")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)

    train_ds = FolderDataset(root / "train" / "good", args.resize, args.imagesize, args.max_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=4)
    test_sets = {sub: FolderDataset(root / "test" / sub, args.resize, args.imagesize, args.n_test)
                 for sub in args.test_subdirs}
    print(f"train {len(train_ds)} images | test sets: "
          + ", ".join(f"{k}={len(v)}" for k, v in test_sets.items()))

    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=patchcore.backbones.load("wideresnet50"),
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        anomaly_score_num_nn=1,
        featuresampler=(patchcore.sampler.IdentitySampler() if args.coreset_percentage >= 1.0
                        else patchcore.sampler.ApproximateGreedyCoresetSampler(
                            percentage=args.coreset_percentage, device=device)),
        nn_method=(TorchNN(device) if args.nn == "torch"
                   else patchcore.common.FaissNN(False, args.faiss_workers)),
    )

    print("fitting coreset ...")
    model.fit(train_dl)

    for sub, test_ds in test_sets.items():
        print(f"scoring {sub} ...")
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, num_workers=4)
        _, masks, _, _ = model.predict(test_dl)

        out_dir = Path(args.out_root) / sub / "maps" / "patchcore"
        out_dir.mkdir(parents=True, exist_ok=True)
        for sid, mask in zip(test_ds.ids, masks):
            m = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
            # PatchCore segments at `imagesize`; the contract is 256x256.
            m = torch.nn.functional.interpolate(
                m, size=(args.target_size, args.target_size),
                mode="bilinear", align_corners=False)[0, 0].numpy().astype(np.float32)
            np.save(out_dir / f"{sid}.npy", m)
        print(f"wrote {len(test_ds.ids)} maps to {out_dir}")


if __name__ == "__main__":
    main()
