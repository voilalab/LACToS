"""Unified full-metric scorer for delta-map dumps (npz: delta_map, gt_mask, valid_mask).

Computes, at eval sigma 5 (project reference operating point), for one dump dir
(or each category subdir with --group_by_subdir):

  Dataset-pooled (all valid pixels of all images scored together):
    pooled_auroc, pooled_aupr, pooled_f1max (one global threshold),
    aupro (region-based, FPR<=0.3), n_regions
  Per-image (metric per image, oracle/best threshold per image, then averaged):
    per_image_auroc, per_image_ap, per_image_f1max,
    per_image_snr, per_image_snr_paper, per_image_mask_mse, psnr_db
  Bookkeeping: n_images (images with both classes), n_maps

Usage:
  python tools/metrics/full_metric_table.py --maps_dir DIR --out OUT.json \
      [--group_by_subdir] [--skip_sub good combined] [--sigma 5.0]
"""
import argparse, glob, json, os
import numpy as np
from scipy.ndimage import label as cc_label, gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score

EIGHT = np.ones((3, 3), dtype=int)


def f1max_counts(scores, labels, bins=8192, lo=None, hi=None):
    lo = scores.min() if lo is None else lo
    hi = scores.max() if hi is None else hi
    h_pos, _ = np.histogram(scores[labels == 1], bins=bins, range=(lo, hi))
    h_neg, _ = np.histogram(scores[labels == 0], bins=bins, range=(lo, hi))
    tp = np.cumsum(h_pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(h_neg[::-1])[::-1].astype(np.float64)
    fn = h_pos.sum() - tp
    return float((2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).max())


def aupro(maps, gts, valids, fpr_limit=0.3, n_thr=600):
    normal = np.concatenate([m[(g == 0) & v] for m, g, v in zip(maps, gts, valids)])
    ns = np.sort(normal); n = len(ns)
    fprs = np.linspace(0, fpr_limit, n_thr + 1)[1:]
    thr = ns[np.clip((n * (1 - fprs)).astype(int), 0, n - 1)]
    fpr_act = 1.0 - np.searchsorted(ns, thr, side="left") / n
    regions = []
    for m, g, v in zip(maps, gts, valids):
        lab, k = cc_label((g > 0) & v, structure=EIGHT)
        for r in range(1, k + 1):
            regions.append(np.sort(m[lab == r].ravel()))
    if not regions:
        return float("nan"), 0
    pro = np.zeros(len(thr))
    for rs in regions:
        pro += 1.0 - np.searchsorted(rs, thr, side="left") / len(rs)
    pro /= len(regions)
    o = np.argsort(fpr_act)
    x = np.concatenate([[0.0], fpr_act[o]])
    y = np.concatenate([[pro[o][0]], pro[o]])
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(y, x) / fpr_limit), len(regions)


def full_metrics(maps, gts, valids):
    scores = np.concatenate([m[v] for m, v in zip(maps, valids)])
    labels = np.concatenate([(g > 0).astype(np.uint8)[v] for g, v in zip(gts, valids)])
    res = {"n_maps": len(maps)}
    if labels.max() > 0 and labels.min() < 1:
        res["pooled_auroc"] = float(roc_auc_score(labels, scores))
        res["pooled_aupr"] = float(average_precision_score(labels, scores))
        res["pooled_f1max"] = f1max_counts(scores, labels)
    res["aupro"], res["n_regions"] = aupro(maps, gts, valids)
    # pooled-pixel SNR / mask-MSE (added 2026-09-12): every valid pixel of the dataset in one pool.
    # SNR on the raw pooled scores (same scores as pooled AUROC); MSE on per-image min-max-normalised
    # maps (the normalisation that defines mask-MSE), pooled over all pixels.
    if labels.max() > 0 and labels.min() < 1:
        ood_p, id_p = scores[labels == 1], scores[labels == 0]
        res["pooled_snr"] = float((ood_p.mean() - id_p.mean()) / (id_p.std() + 1e-8))
        sq, cnt = 0.0, 0
        for m, g, v in zip(maps, gts, valids):
            s_ = m[v]; y_ = (g > 0).astype(np.uint8)[v]
            sn_ = (s_ - s_.min()) / ((s_.max() - s_.min()) + 1e-12)
            sq += float(np.sum((sn_ - y_) ** 2)); cnt += y_.size
        res["pooled_mask_mse"] = sq / max(cnt, 1)

    per = {k: [] for k in ("auroc", "ap", "f1max", "snr", "snr_paper", "mask_mse")}
    for m, g, v in zip(maps, gts, valids):
        y = (g > 0).astype(np.uint8)[v]; s = m[v]
        if y.max() == 0 or y.min() == 1:
            continue
        per["auroc"].append(roc_auc_score(y, s))
        per["ap"].append(average_precision_score(y, s))
        per["f1max"].append(f1max_counts(s, y))
        ood, idp = s[y == 1], s[y == 0]
        per["snr"].append((ood.mean() - idp.mean()) / (idp.std() + 1e-8))
        rng = s.max() - s.min()
        sn = (s - s.min()) / (rng + 1e-12)
        on, inn = sn[y == 1], sn[y == 0]
        per["snr_paper"].append(on.mean() / (inn.mean() + 1e-8))
        per["mask_mse"].append(float(np.mean((sn - y) ** 2)))
    res["n_images"] = len(per["auroc"])
    for k in ("auroc", "ap", "f1max", "snr", "snr_paper", "mask_mse"):
        res[f"per_image_{k if k != 'mask_mse' else 'mask_mse'}"] = (
            float(np.mean(per[k])) if per[k] else float("nan"))
    res["psnr_db"] = (float(np.mean([10 * np.log10(1.0 / max(m, 1e-12)) for m in per["mask_mse"]]))
                      if per["mask_mse"] else float("nan"))
    return res


def load_dir(d, sigma, skip_sub):
    maps, gts, valids = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        stem = os.path.basename(f)
        sub = stem.split("__")[0]
        if any(s in sub for s in skip_sub):
            continue
        z = np.load(f)
        m = np.nan_to_num(z["delta_map"].astype(np.float32), nan=0.0)
        if sigma and sigma > 0:
            m = gaussian_filter(m, sigma, mode="nearest")
        maps.append(m)
        gts.append((z["gt_mask"] > 0).astype(np.uint8))
        # NB: some dumps store `valid_mask` as an EMPTY array (shape (0,)) rather
        # than omitting the key, so test the shape, not just membership —
        # otherwise every map scores zero pixels and full_metrics() dies with
        # "zero-size array to reduction operation maximum".
        v = z["valid_mask"] if "valid_mask" in z else None
        valids.append(v.astype(bool) if v is not None and v.shape == m.shape
                      else np.ones(m.shape, bool))
    return maps, gts, valids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group_by_subdir", action="store_true")
    ap.add_argument("--skip_sub", nargs="*", default=["good", "combined"])
    ap.add_argument("--sigma", type=float, default=5.0)
    args = ap.parse_args()

    out = {}
    if args.group_by_subdir:
        groups = sorted(d for d in os.listdir(args.maps_dir)
                        if os.path.isdir(os.path.join(args.maps_dir, d)))
        for g in groups:
            maps, gts, valids = load_dir(os.path.join(args.maps_dir, g), args.sigma, args.skip_sub)
            if not maps:
                print(f"{g}: NO MAPS"); continue
            out[g] = full_metrics(maps, gts, valids)
            r = out[g]
            print(f"{g:14s} n={r['n_images']:4d} AUPRO={r['aupro']:.4f} "
                  f"pAUROC={r.get('pooled_auroc', float('nan')):.4f} "
                  f"piAUROC={r['per_image_auroc']:.4f} piF1={r['per_image_f1max']:.4f}")
        keys = ("aupro", "pooled_auroc", "pooled_aupr", "pooled_f1max",
                "per_image_auroc", "per_image_ap", "per_image_f1max",
                "per_image_snr", "per_image_snr_paper", "per_image_mask_mse", "psnr_db")
        done = [g for g in out if all(k in out[g] for k in keys)]
        if done:
            out["MEAN"] = {k: float(np.mean([out[g][k] for g in done])) for k in keys}
    else:
        maps, gts, valids = load_dir(args.maps_dir, args.sigma, args.skip_sub)
        if not maps:
            raise SystemExit(f"no npz maps in {args.maps_dir}")
        out = full_metrics(maps, gts, valids)
        print(json.dumps(out, indent=1))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
