"""Compute the SLIC mask of every test image before scoring starts, so that concurrent LACToS
scorers of one dataset never race on the same mask.png.

    python toy/prepare_superpixels.py --image_dir toy/data/quad_L1/test/q1_square --figures_root results/toy/quad/L1/figures [--sids 000 001]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toy.score_from_recons import ensure_superpixel_mask   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--figures_root", required=True)
    p.add_argument("--sids", nargs="*", default=None)
    a = p.parse_args()
    image_dir = Path(a.image_dir)
    sids = a.sids or sorted(q.stem for q in image_dir.glob("*.png"))
    for sid in sids:
        ensure_superpixel_mask(Path(a.figures_root) / f"samples_{sid}", image_dir / f"{sid}.png")
    print(f"{len(sids)} masks under {a.figures_root}")


if __name__ == "__main__":
    main()
