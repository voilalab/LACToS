"""Toy experiment 1 (paper Figure 2 and Appendix Figure 7): false positives on the other objects
against the inlier vocabulary, at a fixed detection rate of the Q1 square.

For every map the threshold is set per image at the TPR-th quantile of its anomaly pixels, so
every method and image sits at the same detection rate; the false-positive rate is the share of
the OTHER three objects' pixels above it, averaged over images. Anomaly-free test images carry
no anomaly pixels, so each is thresholded at the median of the anomalous images' thresholds
(variant `per_image_normal`, the paper's). The pooled variants (one global threshold over the
pixels of all images) are reported in the markdown table for comparison. CIs: bootstrap over
images (anomalous and normal strata resampled separately).

Maps are read in the equalized convention: LACToS maps are blurred with sigma 4 first, which is
the blur DDAD and PatchCore apply inside their own pipelines.

Series: DDAD 1 sample (`single`), DDAD 20 sample mean (`mean20`), LACToS 1/5/10/20 samples
(`set1`..`set20`), PatchCore. Layout under --results_root (what toy/run_toy1.sh writes):
    L<N>/<cell>/<mode>/maps/<method>/<id>.npy          anomalous images (test/q1_square)
    L<N>/good/<cell>/<mode>/maps/<method>/<id>.npy     anomaly-free images (test/good)
    L<N>/patchcore/{q1_square,good}/maps/patchcore/<id>.npy

Writes, under --out (default --results_root): toy1_fpr_<cell>.json, toy1_fpr_<cell>.md and, per
TPR, toy1_fpr_<cell>_tpr<TPR>.png (labelled) and toy1_fpr_<cell>_tpr<TPR>_bare.png (the panel
of the paper, no text).

    python toy/analyze_toy1.py [--cell additive_dps_s0.30] [--tprs 0.8 0.9 0.95 0.99]
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
LEVELS = [1, 2, 3, 4]
FG_THRESHOLD = 8                      # grey level separating the background from the objects
SERIES = [("ddad", "single"), ("ddad", "mean20"),
          ("lactos", "set1"), ("lactos", "set5"), ("lactos", "set10"), ("lactos", "set20"),
          ("patchcore", "-")]
N_GRID = 4001                         # threshold grid for the pooled bootstrap: TPR resolution 1/4000
STYLE = {("ddad", "single"): ("#f4978e", "o-", "DDAD 1 sample"), ("ddad", "mean20"): ("#c1121f", "s-", "DDAD 20 sample mean"),
         ("lactos", "set1"): ("#bdd7f0", "v-", "LACToS 1 sample"), ("lactos", "set5"): ("#7fb2e0", "v-", "LACToS 5 samples"),
         ("lactos", "set10"): ("#3b7fc4", "v-", "LACToS 10 samples"), ("lactos", "set20"): ("#10375c", "v-", "LACToS 20 samples"),
         ("patchcore", "-"): ("#52514e", "d-", "PatchCore")}
VARIANT_TITLES = {"per_image": "anomalous images only, per-image threshold",
                  "pooled": "anomalous images only, pooled pixels, one threshold",
                  "per_image_normal": "anomalous + normal images, per-image threshold (the paper)",
                  "pooled_normal": "anomalous + normal images, pooled pixels, one threshold"}
_CFG: Dict = {}                       # results_root, data_root; set before the worker pool forks


def maps_dir(level: int, cell: str, method: str, mode: str, good: bool) -> Path:
    base = Path(_CFG["results_root"]) / f"L{level}"
    if method == "patchcore":
        return base / "patchcore" / ("good" if good else "q1_square") / "maps" / "patchcore"
    return base / ("good" if good else "") / cell / mode / "maps" / method


def load_image_set(level: int, cell: str, method: str, mode: str, good: bool) -> Optional[List[Dict[str, np.ndarray]]]:
    """Per image: finite scores split into anomaly / other-object / background pixels."""
    d = maps_dir(level, cell, method, mode, good)
    if not d.is_dir():
        return None
    data = Path(_CFG["data_root"]) / f"quad_L{level}"
    sub = "good" if good else "q1_square"
    out = []
    for f in sorted(d.glob("*.npy")):
        sid = f.stem
        m = np.load(f).astype(np.float64)
        if method == "lactos":
            m = gaussian_filter(np.nan_to_num(m, nan=np.nanmin(m)), sigma=4.0, mode="nearest")
        finite = np.isfinite(m)
        x = np.array(Image.open(data / f"test/{sub}/{sid}.png").convert("L")).astype(float)
        if good:
            gt = np.zeros(m.shape, dtype=bool)
        else:
            gt = np.array(Image.open(data / f"ground_truth/{sub}/{sid}_mask.png")) > 0
            if not (finite & gt).any():
                continue
        fg = x > FG_THRESHOLD
        out.append({"sid": sid, "pos": m[finite & gt], "obj": m[finite & fg & ~gt], "bg": m[finite & ~fg & ~gt]})
    return out or None


def _ge_counts(arr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    s = np.sort(arr)
    return s.size - np.searchsorted(s, grid, side="left")


def per_image_stats(anom: List[Dict], good: List[Dict], tpr: float, n_boot: int, seed: int) -> Dict:
    """Per-image threshold at the TPR-th anomaly quantile, FPR averaged over images; normal images are
    thresholded at the median of the anomalous images' thresholds."""
    thrs = np.asarray([np.quantile(a["pos"], 1.0 - tpr) for a in anom])
    thr_norm = float(np.median(thrs))
    obj = [(a["obj"] >= t).mean() if a["obj"].size else np.nan for a, t in zip(anom, thrs)]
    bg = [(a["bg"] >= t).mean() if a["bg"].size else np.nan for a, t in zip(anom, thrs)]
    obj_g = [(g["obj"] >= thr_norm).mean() if g["obj"].size else np.nan for g in good]
    bg_g = [(g["bg"] >= thr_norm).mean() if g["bg"].size else np.nan for g in good]
    obj_a, obj_g = np.asarray(obj), np.asarray(obj_g)
    all_obj = np.concatenate([obj_a, obj_g])
    rng = np.random.default_rng(seed)
    na, ng = obj_a.size, obj_g.size
    da = obj_a[rng.integers(0, na, size=(n_boot, na))]
    means = (np.nansum(da, axis=1) + (np.nansum(obj_g[rng.integers(0, ng, size=(n_boot, ng))], axis=1) if ng else 0)) / (na + ng)
    res = {"fpr_objects": float(np.nanmean(all_obj)),
           "fpr_objects_lo": float(np.percentile(means, 2.5)), "fpr_objects_hi": float(np.percentile(means, 97.5)),
           "fpr_background": float(np.nanmean(np.concatenate([bg, bg_g]))), "n_anom": len(anom), "n_normal": len(good)}
    if good:
        res["threshold_normal"] = thr_norm
        res["fpr_normal_images"] = float(np.nanmean([(np.concatenate([g["obj"], g["bg"]]) >= thr_norm).mean() for g in good]))
        res["fpr_normal_objects"] = float(np.nanmean(obj_g))
    return res


def pooled_stats(anom: List[Dict], good: List[Dict], tpr: float, n_boot: int, seed: int) -> Dict:
    """One threshold over the pooled anomaly pixels; FPR over the pooled negatives; bootstrap over images."""
    pos = np.concatenate([a["pos"] for a in anom])
    thr = float(np.quantile(pos, 1.0 - tpr))
    imgs = anom + good
    obj = np.concatenate([i["obj"] for i in imgs])
    bg = np.concatenate([i["bg"] for i in imgs])
    neg = np.concatenate([obj, bg])
    res = {"threshold": thr, "fpr_objects": float((obj >= thr).mean()), "fpr_background": float((bg >= thr).mean()),
           "fpr_all": float((neg >= thr).mean()), "n_anom": len(anom), "n_normal": len(good),
           "n_pos_px": int(pos.size), "n_obj_px": int(obj.size), "n_bg_px": int(bg.size)}
    tprs = np.asarray([(a["pos"] >= thr).mean() for a in anom])
    res["tpr_per_image_mean"] = float(tprs.mean()); res["tpr_per_image_min"] = float(tprs.min())
    res["images_below_half_tpr"] = int((tprs < 0.5).sum())
    if good:
        gi = np.asarray([(np.concatenate([g["obj"], g["bg"]]) >= thr).mean() for g in good])
        res["fpr_normal_images"] = float(gi.mean())
        res["fpr_normal_objects"] = float(np.concatenate([g["obj"] for g in good]).__ge__(thr).mean())
    labels = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)])
    scores = np.concatenate([pos, neg])
    res["auroc"] = float(roc_auc_score(labels, scores)); res["ap"] = float(average_precision_score(labels, scores))
    grid = np.quantile(pos, np.linspace(0.0, 1.0, N_GRID))
    P = np.stack([_ge_counts(a["pos"], grid) for a in anom]).astype(np.float64)
    Ptot = np.asarray([a["pos"].size for a in anom], dtype=np.float64)
    Oa = np.stack([_ge_counts(a["obj"], grid) for a in anom]).astype(np.float64)
    Oa_tot = np.asarray([a["obj"].size for a in anom], dtype=np.float64)
    rng = np.random.default_rng(seed)
    na = len(anom)
    Wa = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(n_boot)]).astype(np.float64)
    pos_ge = Wa @ P
    need = tpr * (Wa @ Ptot)
    t = np.asarray([np.searchsorted(-row, -n, side="right") - 1 for row, n in zip(pos_ge, need)])
    t = np.clip(t, 0, N_GRID - 1)
    fp = (Wa @ Oa)[np.arange(n_boot), t]
    tot = Wa @ Oa_tot
    if good:
        Og = np.stack([_ge_counts(g["obj"], grid) for g in good]).astype(np.float64)
        Og_tot = np.asarray([g["obj"].size for g in good], dtype=np.float64)
        ng = len(good)
        Wg = np.stack([np.bincount(rng.integers(0, ng, ng), minlength=ng) for _ in range(n_boot)]).astype(np.float64)
        fp = fp + (Wg @ Og)[np.arange(n_boot), t]
        tot = tot + Wg @ Og_tot
    boot = fp / np.maximum(tot, 1)
    res["fpr_objects_lo"] = float(np.percentile(boot, 2.5)); res["fpr_objects_hi"] = float(np.percentile(boot, 97.5))
    return res


def _job(args) -> Optional[Dict]:
    level, cell, method, mode, tprs, n_boot, seed = args
    anom = load_image_set(level, cell, method, mode, good=False)
    if anom is None:
        return None
    good = load_image_set(level, cell, method, mode, good=True)
    out = {"level": level, "cell": cell, "method": method, "mode": mode, "variants": {}}
    for tpr in tprs:
        key = f"{tpr:.2f}"
        out["variants"][key] = {"per_image": per_image_stats(anom, [], tpr, n_boot, seed),
                                "pooled": pooled_stats(anom, [], tpr, n_boot, seed)}
        if good:
            out["variants"][key]["per_image_normal"] = per_image_stats(anom, good, tpr, n_boot, seed)
            out["variants"][key]["pooled_normal"] = pooled_stats(anom, good, tpr, n_boot, seed)
    return out


def _panel(ax, by, series, tpr, variant):
    for m, mo in series:
        col, st, _ = STYLE.get((m, mo), ("#888888", "x-", f"{m} {mo}"))
        pts = [(l, by.get((m, mo, l), {}).get("variants", {}).get(f"{tpr:.2f}", {}).get(variant)) for l in LEVELS]
        pts = [(l, x) for l, x in pts if x]                  # the levels present (all four in a full run)
        if not pts:
            continue
        ls, v = [l for l, _ in pts], [x for _, x in pts]
        ax.plot(ls, [x["fpr_objects"] for x in v], st, color=col, markersize=5, linewidth=1.7 if m != "patchcore" else 1.4,
                label=STYLE.get((m, mo), (None, None, f"{m} {mo}"))[2])
        if "fpr_objects_lo" in v[0]:
            ax.fill_between(ls, [x["fpr_objects_lo"] for x in v], [x["fpr_objects_hi"] for x in v], color=col, alpha=0.13, linewidth=0)
    ax.set_xticks(LEVELS); ax.set_xlim(LEVELS[0] - 0.15, LEVELS[-1] + 0.15)
    ax.grid(alpha=0.25, linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#e6e5e1")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", default=str(ROOT / "results" / "toy" / "quad"))
    p.add_argument("--data_root", default=str(ROOT / "toy" / "data"))
    p.add_argument("--out", default=None, help="default: --results_root")
    p.add_argument("--cell", default="additive_dps_s0.30", help="reconstruction cell (the paper: DPS sigma 0.30)")
    p.add_argument("--tprs", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99])
    p.add_argument("--variant", default="per_image_normal", choices=list(VARIANT_TITLES), help="variant drawn in the figures")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    _CFG.update(results_root=args.results_root, data_root=args.data_root)
    out = Path(args.out or args.results_root); out.mkdir(parents=True, exist_ok=True)
    tag = args.cell.replace(".", "")

    jobs = [(L, args.cell, m, mo, args.tprs, args.n_boot, 0) for L in LEVELS for m, mo in SERIES]
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = [r for r in pool.map(_job, jobs) if r]
    if not rows:
        raise SystemExit(f"no maps under {args.results_root} for cell {args.cell}")
    by = {(r["method"], r["mode"], r["level"]): r for r in rows}
    have_normal = any("pooled_normal" in v for r in rows for v in r["variants"].values())
    variants = ["per_image", "pooled"] + (["per_image_normal", "pooled_normal"] if have_normal else [])
    (out / f"toy1_fpr_{tag}.json").write_text(json.dumps(
        {"cell": args.cell, "tprs": args.tprs, "n_boot": args.n_boot, "series": [f"{m}:{mo}" for m, mo in SERIES], "rows": rows}, indent=2))

    label = lambda m, mo: STYLE.get((m, mo), (None, None, f"{m} {mo}"))[2]
    md = ["# Toy experiment 1: false positives on the other objects vs. inlier vocabulary", "",
          f"Cell `{args.cell}`, equalized maps, {max(r['variants'][f'{args.tprs[0]:.2f}']['per_image']['n_anom'] for r in rows)} anomalous "
          f"images per level" + (f", {max(v.get('pooled_normal', {}).get('n_normal', 0) for r in rows for v in r['variants'].values())} normal images per level" if have_normal else "")
          + f". FPR objects = share of the other three objects' pixels above the threshold; "
          f"95% bootstrap CIs over images ({args.n_boot} resamples). Written by `toy/analyze_toy1.py`.", ""]
    for tpr in args.tprs:
        key = f"{tpr:.2f}"
        md += [f"## TPR = {tpr:.0%}", ""]
        for var in variants:
            md += [f"### {VARIANT_TITLES[var]}", "", "| series | " + " | ".join(f"L{l}" for l in LEVELS) + " | L4 - L1 |",
                   "|---|" + "---|" * (len(LEVELS) + 1)]
            for m, mo in SERIES:
                v = [by.get((m, mo, l), {}).get("variants", {}).get(key, {}).get(var) for l in LEVELS]
                if not any(v):
                    continue
                md.append(f"| {label(m, mo)} | " + " | ".join(
                    f"{x['fpr_objects']:.3f} [{x['fpr_objects_lo']:.3f}, {x['fpr_objects_hi']:.3f}]" if x else "-" for x in v)
                    + (f" | {v[-1]['fpr_objects'] - v[0]['fpr_objects']:+.3f} |" if v[0] and v[-1] else " | - |"))
            md.append("")
    (out / f"toy1_fpr_{tag}.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variant = args.variant if args.variant in variants else "per_image"
    top = max(v["variants"][f"{t:.2f}"][variant].get("fpr_objects_hi", v["variants"][f"{t:.2f}"][variant]["fpr_objects"])
              for v in rows for t in args.tprs if variant in v["variants"][f"{t:.2f}"])
    ymax = float(np.ceil(top * 10 + 1e-9) / 10)
    for tpr in args.tprs:
        t = int(round(tpr * 100))
        # labelled panel
        fig, ax = plt.subplots(figsize=(5.6, 4.2), facecolor="#ffffff")
        _panel(ax, by, SERIES, tpr, variant)
        ax.set_ylim(0, ymax); ax.set_yticks(np.round(np.arange(0.0, ymax + 1e-9, 0.1), 2))
        ax.set_xticklabels([f"Level {l}" for l in LEVELS]); ax.set_xlabel("inlier complexity")
        ax.set_ylabel(f"FPR @ TPR = {tpr:.0%}")
        ax.set_title(VARIANT_TITLES[variant], fontsize=8.5, color="#52514e")
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.09, 1, 1]); fig.savefig(out / f"toy1_fpr_{tag}_tpr{t:02d}.png", dpi=args.dpi, facecolor="#ffffff"); plt.close(fig)
        # bare panel (the paper's)
        aw, ah, mg = 3.873, 2.173, 0.12
        fig = plt.figure(figsize=(aw + 2 * mg, ah + 2 * mg), facecolor="#ffffff")
        ax = fig.add_axes([mg / (aw + 2 * mg), mg / (ah + 2 * mg), aw / (aw + 2 * mg), ah / (ah + 2 * mg)])
        _panel(ax, by, SERIES, tpr, variant)
        ax.set_ylim(0, ymax); ax.set_yticks(np.round(np.arange(0.0, ymax + 1e-9, 0.1), 2))
        ax.tick_params(labelbottom=False, labelleft=False, width=1.6, length=5, color="#52514e")
        fig.savefig(out / f"toy1_fpr_{tag}_tpr{t:02d}_bare.png", dpi=args.dpi, facecolor="#ffffff"); plt.close(fig)
        print(f"wrote {out / f'toy1_fpr_{tag}_tpr{t:02d}.png'} (+ _bare)")
    print(f"wrote {out / f'toy1_fpr_{tag}.json'}, {out / f'toy1_fpr_{tag}.md'}")


if __name__ == "__main__":
    main()
