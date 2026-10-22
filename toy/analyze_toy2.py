"""Toy experiment 2 (paper Appendix Figure 9): does a detector flag an in-distribution but rare
feature, and does it still single out the true anomaly, as the square probability p falls?

Every prior (p = 0.5 .. 0.1) is scored on the same 50 test images, each with a square at Q1 (the
anomaly) and at least one identical square at Q2-Q4 (in-distribution, its "twins"). Two
pooled-pixel AUROCs are computed over all images, with a bootstrap over images for the CI:

  id   in-distribution square vs. in-distribution circle: pixels of the Q2-Q4 squares (positive)
       against the Q2-Q4 circles (negative). 0.5 = a normal square is scored like a circle;
       1.0 = the detector calls every normal square anomalous.
  q1   inlier vs. outlier square: the Q1 square's pixels (positive) against its twins' pixels
       (negative). 1.0 = the anomaly is singled out from identical in-distribution squares.

Series: LACToS with K samples (`set<K>`, K in --ks), DDAD 1 sample (`single`) and DDAD 20 sample
mean (`mean20`, id panel). Maps are read in the equalized convention (LACToS blurred with sigma 4).
Layout under --results_root (toy/run_toy2.sh): <dataset>/<cell>/<mode>/maps/<method>/<id>.npy.
The object masks are regenerated from the test-set seeds (toy/gen_data.py).

Writes, under --out (default --results_root): toy2_auroc_<cell>.json, toy2_auroc_<cell>.md,
toy2_auroc_<cell>.png (labelled, both panels) and toy2_auroc_<cell>_{id,q1}_bare.png (the panels
of the paper, no text).

    python toy/analyze_toy2.py [--cell additive_dps_s0.10] [--ks 1 2 5 10 20 40]
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from toy.gen_data import ANCHORS, P_SWEEP, load_config, make_ood_plus   # noqa: E402
from toy.shapes import Circle, Square                                   # noqa: E402

DATASETS = [("quadp_p50", 0.5), ("quadp_p40", 0.4), ("quadp_p30", 0.3), ("quadp_p20", 0.2), ("quadp_p10", 0.1)]
N_TEST = P_SWEEP["n_test"]
K_COL = {1: "#c9dff5", 2: "#a5c8ee", 5: "#7fb2e0", 10: "#4f92d3", 20: "#2a78d6", 40: "#10375c"}
_CFG: Dict = {}
_MASKS: Dict[str, Dict] = {}


def object_masks(sid: str) -> Dict:
    """Regenerate test image <sid>: ground truth (the Q1 square) and the coverage >= 0.5 masks of
    every in-distribution square (twin) and circle."""
    cfg = load_config()
    size, ss = int(cfg["image_size"]), int(cfg["supersample"])
    img, gt, meta = make_ood_plus(P_SWEEP["test_seed"] + int(sid), cfg)
    twins, circles = [], []
    for o in meta["objects"]:
        if o["kind"] == "Square":
            twins.append(Square(o["cx"], o["cy"], o["side"], o["theta"], o["intensity"]))
        else:
            circles.append(Circle(o["cx"], o["cy"], o["r"], o["intensity"]))
    cov = lambda s: s.coverage(size, ss) >= 0.5
    return {"image": img, "gt": gt > 0, "twins": [cov(s) for s in twins], "circles": [cov(s) for s in circles]}


def map_dir(ds: str, cell: str, method: str, kind: str, K: int) -> Path:
    base = Path(_CFG["results_root"]) / ds
    if method == "patchcore":
        return base / "patchcore" / P_SWEEP["test_variant"] / "maps" / "patchcore"
    if method == "ddad":
        sub = "single" if K == 1 else f"mean{K}"
    else:
        sub = "set1" if K == 1 else f"{kind}{K}"   # one draw: LACToS's mean of 1 and set of 1 coincide
    return base / cell / sub / "maps" / method


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _job(args) -> Optional[Dict]:
    ds, cell, method, kind, K, n_boot = args
    d = map_dir(ds, cell, method, kind, K)
    sids = sorted(f.stem for f in d.glob("*.npy")) if d.is_dir() else []   # every scored image (all 50 in a full run)
    if not sids:
        return None
    per = {"id": [], "q1": []}                  # per-image pixel pools, so the bootstrap can resample images
    for s in sids:
        a = np.load(d / f"{s}.npy").astype(np.float64)
        if method == "lactos":
            a = gaussian_filter(np.nan_to_num(a, nan=np.nanmin(a)), sigma=4.0, mode="nearest")
        m, fin = _MASKS[s], np.isfinite(a)
        twins = np.any(m["twins"], axis=0) & fin
        circ = np.any(m["circles"], axis=0) & fin if m["circles"] else np.zeros_like(fin)
        per["id"].append((a[twins], a[circ]))
        per["q1"].append((a[m["gt"] & fin], a[twins]))
    out = {"dataset": ds, "cell": cell, "method": method, "kind": kind, "K": K, "n_images": len(sids)}
    rng = np.random.default_rng(0)
    for key, pairs in per.items():
        pooled = _auroc(np.concatenate([p for p, _ in pairs]), np.concatenate([n for _, n in pairs]))
        per_img = [v for v in (_auroc(p, n) for p, n in pairs) if np.isfinite(v)]
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(pairs), len(pairs))
            boots.append(_auroc(np.concatenate([pairs[i][0] for i in idx]), np.concatenate([pairs[i][1] for i in idx])))
        out[f"{key}_pooled"] = pooled
        out[f"{key}_pooled_lo"], out[f"{key}_pooled_hi"] = float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))
        out[f"{key}_perimage"] = float(np.mean(per_img)) if per_img else float("nan")
    return out


def _panel(ax, by, key: str, cell: str, ks: List[int], ps: List[float], ds_of: Dict, with_patchcore: bool):
    series = lambda method, kind, K: [by.get((ds_of[pv], cell, method, kind, K)) for pv in ps]
    for K in ks:
        v = series("lactos", "set", K)
        if not all(v):
            continue
        ax.plot(ps, [r[f"{key}_pooled"] for r in v], "v-", color=K_COL.get(K, "#2a78d6"), linewidth=1.8, markersize=5,
                label=f"LACToS {K} sample{'s' if K > 1 else ''}")
        if K == max(ks):
            ax.fill_between(ps, [r[f"{key}_pooled_lo"] for r in v], [r[f"{key}_pooled_hi"] for r in v],
                            color=K_COL.get(K, "#2a78d6"), alpha=0.12, linewidth=0)
    v = series("ddad", "mean", 1)
    if all(v):
        col = "#f4978e" if key == "id" else "#c2410c"
        ax.plot(ps, [r[f"{key}_pooled"] for r in v], "o-", color=col, linewidth=1.8, markersize=5, label="DDAD 1 sample")
        ax.fill_between(ps, [r[f"{key}_pooled_lo"] for r in v], [r[f"{key}_pooled_hi"] for r in v], color=col, alpha=0.15, linewidth=0)
    v = series("ddad", "mean", 20)
    if key == "id" and all(v):
        ax.plot(ps, [r[f"{key}_pooled"] for r in v], "s-", color="#c1121f", linewidth=1.8, markersize=5, label="DDAD 20 sample mean")
    pc = [by.get((ds_of[pv], "-", "patchcore", "-", 0)) for pv in ps]
    if with_patchcore and all(pc):
        ax.plot(ps, [r[f"{key}_pooled"] for r in pc], "d-", color="#4b5563", linewidth=1.6, markersize=5, label="PatchCore")
    ax.axhline(0.5, color="#52514e", linewidth=0.8, alpha=0.6)
    ax.set_xticks(ps); ax.set_xlim(max(ps) + 0.05, min(ps) - 0.05)       # p = 0.5 leftmost
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=10, width=1.6, length=5, color="#52514e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#e6e5e1")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", default=str(ROOT / "results" / "toy" / "quadp"))
    p.add_argument("--data_root", default=str(ROOT / "toy" / "data"), help="checks the regenerated test image against the stored one")
    p.add_argument("--out", default=None, help="default: --results_root")
    p.add_argument("--cell", default="additive_dps_s0.10", help="reconstruction cell (the paper: DPS sigma 0.10)")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40])
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--with_patchcore", action="store_true", help="add PatchCore maps (<dataset>/patchcore/...) when present")
    p.add_argument("--ymin", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    _CFG.update(results_root=args.results_root)
    out = Path(args.out or args.results_root); out.mkdir(parents=True, exist_ok=True)
    tag = args.cell.replace(".", "")

    for i in range(N_TEST):
        _MASKS[f"{i:03d}"] = object_masks(f"{i:03d}")          # forked workers inherit these
    stored = Path(args.data_root) / DATASETS[0][0] / "test" / P_SWEEP["test_variant"] / "000.png"
    if stored.exists():
        assert np.array_equal(_MASKS["000"]["image"], np.array(Image.open(stored))[..., 0]), \
            f"regenerated test image differs from {stored}"
    jobs = [(ds, args.cell, "lactos", "set", K, args.n_boot) for ds, _ in DATASETS for K in args.ks]
    jobs += [(ds, args.cell, "ddad", "mean", K, args.n_boot) for ds, _ in DATASETS for K in (1, 20)]
    jobs += [(ds, "-", "patchcore", "-", 0, args.n_boot) for ds, _ in DATASETS]
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = [r for r in pool.map(_job, jobs) if r]
    if not rows:
        raise SystemExit(f"no complete map sets under {args.results_root} for cell {args.cell}")
    (out / f"toy2_auroc_{tag}.json").write_text(json.dumps({"cell": args.cell, "ks": args.ks, "n_boot": args.n_boot, "rows": rows}, indent=2))

    by = {(r["dataset"], r["cell"], r["method"], r["kind"], r["K"]): r for r in rows}
    ps = [pv for _, pv in DATASETS]; ds_of = {pv: ds for ds, pv in DATASETS}
    fmt = lambda r, key: ("-" if not r or not np.isfinite(r[f"{key}_pooled"]) else
                          f"{r[f'{key}_pooled']:.3f} [{r[f'{key}_pooled_lo']:.3f}, {r[f'{key}_pooled_hi']:.3f}]")
    series = [("DDAD 1 sample", "ddad", "mean", 1), ("DDAD 20 sample mean", "ddad", "mean", 20)] + \
             [(f"LACToS {K} sample{'s' if K > 1 else ''}", "lactos", "set", K) for K in args.ks]
    md = ["# Toy experiment 2: pooled-pixel AUROC against the square probability p", "",
          f"Cell `{args.cell}`, equalized maps, {max(r['n_images'] for r in rows)} test images per prior; 95% bootstrap CIs over images "
          f"({args.n_boot} resamples). Written by `toy/analyze_toy2.py`.", ""]
    for key, title in (("id", "In-distribution square vs. in-distribution circle (0.5 = a normal square scored like a circle)"),
                       ("q1", "Inlier vs. outlier square: the Q1 square vs. its in-distribution twins (1 = the anomaly is singled out)")):
        md += [f"## {title}", "", "| detector | " + " | ".join(f"p = {pv}" for pv in ps) + " |", "|---|" + "---|" * len(ps)]
        for label, m, k, K in series:
            v = [by.get((ds_of[pv], args.cell, m, k, K)) for pv in ps]
            if any(v):
                md.append(f"| {label} | " + " | ".join(fmt(r, key) for r in v) + " |")
        pc = [by.get((ds_of[pv], "-", "patchcore", "-", 0)) for pv in ps]
        if any(pc):
            md.append("| PatchCore | " + " | ".join(fmt(r, key) for r in pc) + " |")
        md.append("")
    (out / f"toy2_auroc_{tag}.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(K for K in args.ks if any((ds, args.cell, "lactos", "set", K) in by for ds, _ in DATASETS))
    titles = {"id": "Inlier square vs. inlier circle", "q1": "Inlier vs. outlier square"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), facecolor="#ffffff", sharey=True)
    for ax, key in zip(axes, ("id", "q1")):
        _panel(ax, by, key, args.cell, ks, ps, ds_of, args.with_patchcore)
        ax.set_ylim(args.ymin, 1.02); ax.set_yticks(np.round(np.arange(args.ymin, 1.001, 0.1), 2))
        ax.set_xticklabels([f"Level {i + 1}\n(p = {pv})" for i, pv in enumerate(ps)], fontsize=8)
        ax.set_title(titles[key], fontsize=11)
    axes[0].set_ylabel("pooled-pixel AUROC")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out / f"toy2_auroc_{tag}.png", dpi=args.dpi, facecolor="#ffffff"); plt.close(fig)
    for key in ("id", "q1"):
        fig, ax = plt.subplots(figsize=(5.2, 4.3), facecolor="#ffffff")
        _panel(ax, by, key, args.cell, ks, ps, ds_of, args.with_patchcore)
        ax.set_ylim(args.ymin, 1.02); ax.set_yticks(np.round(np.arange(args.ymin, 1.001, 0.1), 2))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        fig.tight_layout(); fig.savefig(out / f"toy2_auroc_{tag}_{key}_bare.png", dpi=args.dpi, facecolor="#ffffff"); plt.close(fig)
    print(f"wrote {out / f'toy2_auroc_{tag}.json'}, {out / f'toy2_auroc_{tag}.md'}, {out / f'toy2_auroc_{tag}.png'} (+ _id_bare, _q1_bare)")


if __name__ == "__main__":
    main()
