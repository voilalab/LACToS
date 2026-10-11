#!/usr/bin/env python
"""Stage 4: build Table 1 from the stage-2 (LACToS) and stage-3 (baseline) outputs and
compare every cell with the paper (expected/table1.json).

  python scripts/40_make_table1.py                 # all 7 methods x 3 benchmarks
  python scripts/40_make_table1.py --only lactos   # LACToS column only (after scripts/20_score_lactos.sh)
  python scripts/40_make_table1.py --only baselines

Writes results/table1.json, results/table1.md, results/table1.tex and prints the diff
against the paper. Exit code 0 only when every available cell agrees to 4 decimals.

Metrics (tools/metrics via full_metric_table.full_metrics): all valid pixels of all test
images of a benchmark are pooled; AUROC and AP (average precision) over that pool; AUPRO is
the area under the per-region-overlap curve up to FPR 0.3, normalised by 0.3.
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from full_metric_table import full_metrics            # noqa: E402

OUT = os.environ.get("OUT", os.path.join(ROOT, "results"))
LMAPS = os.path.join(OUT, "lactos_maps")
BMAPS = os.path.join(OUT, "baseline_maps")
EXPECTED = json.load(open(os.path.join(ROOT, "expected", "table1.json")))
BENCHES = ["CT", "ffhq", "xray"]
NAMES = {"CT": "CHAOS", "ffhq": "FFHQ", "xray": "DVXray"}
METHODS = ["patchcore", "simplenet", "supersimplenet", "efficientad", "glass", "ddad", "lactos"]
LABELS = {"patchcore": "PatchCore", "simplenet": "SimpleNet", "supersimplenet": "SuperSimpleNet",
          "efficientad": "EfficientAD", "glass": "GLASS", "ddad": "DDAD", "lactos": "LACToS (Ours)"}
N_IMAGES = {"CT": 200, "ffhq": 100, "xray": 144}

# LACToS operating points (paper Table 5): dump tags produced by scripts/20_score_lactos.sh
LACTOS = {
    "CT":   dict(full="sp100_m5_hs0.5_b16x64", rgb=None, lam=1.0, sigma=2, sets=["cta", "ctn"]),
    "ffhq": dict(full="sp20_m20_hs1_b16x32", rgb="sp20_m20_hs1_rgbonly_b16", lam=0.25, sigma=5, sets=["faces", "moreface", "faces_normal"]),
    "xray": dict(full=None, rgb="sp10_m20_hs0_rgbonly_b16", lam=0.0, sigma=5, sets=["xrsc", "xrgd"]),
}
# baseline picks (paper Table 6 + published blur): setting dir and post-hoc knob
BASELINE = {
    "patchcore":      {b: ("default", 1) for b in BENCHES},
    "simplenet":      {b: ("default", None) for b in BENCHES},
    "supersimplenet": {b: ("default", None) for b in BENCHES},
    "glass":          {b: ("default", None) for b in BENCHES},
    "efficientad":    {"CT": ("nopad", 0.0), "ffhq": ("default", 0.7), "xray": ("nopad", 0.2)},
    "ddad":           {"CT": ("w2_T250_s25_e1", 1.0), "ffhq": ("w2_T250_s25_e1", 4.0), "xray": ("w4_T250_s25_e1", 4.0)},
}


def _load_dumps(bench, tag, sets):
    files = [f for s in sets for f in sorted(glob.glob(os.path.join(LMAPS, bench, tag, s, "samples_*.npz")))]
    return [np.load(f) for f in files]


def _valid(z):
    v = z["valid_mask"]
    return v.astype(bool) if v.shape == z["delta_map"].shape else np.ones(z["delta_map"].shape, bool)


def lactos_cell(bench):
    """Pooled AUROC / AP / AUPRO of LACToS at the paper's operating point.
    score = d_rgb + lam * (d_full - d_rgb): lam=1 -> full dump only, lam=0 -> RGB-only dump only."""
    op = LACTOS[bench]
    full = _load_dumps(bench, op["full"], op["sets"]) if op["lam"] > 0 else None
    rgb = _load_dumps(bench, op["rgb"], op["sets"]) if op["lam"] < 1 else None
    base = full if full is not None else rgb
    if not base:
        return None
    n = len(base)
    if n != N_IMAGES[bench]:
        print(f"  [warn] {bench}: {n} LACToS dumps, expected {N_IMAGES[bench]} (run scripts/20_score_lactos.sh {bench})")
    maps, gts, valids = [], [], []
    for i in range(n):
        z = base[i]
        d = np.nan_to_num(z["delta_map"].astype(np.float64))
        if full is not None and rgb is not None:
            r = np.nan_to_num(rgb[i]["delta_map"].astype(np.float64))
            d = r + op["lam"] * (d - r)
        d = d.astype(np.float32)
        maps.append(gaussian_filter(d, op["sigma"], mode="nearest") if op["sigma"] > 0 else d)
        gts.append((z["gt_mask"] > 0).astype(np.uint8)); valids.append(_valid(z))
    r = full_metrics(maps, gts, valids)
    return [r["pooled_auroc"], r["pooled_aupr"], r["aupro"]]


def baseline_cell(bench, method):
    sys.path.insert(0, os.path.join(ROOT, "baselines"))
    from compose import pub_map                     # noqa: E402
    setting, knob = BASELINE[method][bench]
    files = sorted(glob.glob(os.path.join(BMAPS, bench, method, setting, "*.npz")))
    if not files:
        return None
    if len(files) != N_IMAGES[bench]:
        print(f"  [warn] {bench}/{method}: {len(files)} maps, expected {N_IMAGES[bench]} (run scripts/30_run_baselines.sh {bench})")
    maps, gts, valids = [], [], []
    for f in files:
        z = dict(np.load(f))
        maps.append(pub_map(method, bench, z, knob))
        gts.append(z["gt_mask"].astype(np.uint8)); valids.append(z["valid_mask"].astype(bool))
    r = full_metrics(maps, gts, valids)
    return [r["pooled_auroc"], r["pooled_aupr"], r["aupro"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["lactos", "baselines", "all"], default="all")
    a = ap.parse_args()
    methods = [m for m in METHODS if a.only == "all" or (m == "lactos") == (a.only == "lactos")]
    table = {b: {} for b in BENCHES}
    for b in BENCHES:
        for m in methods:
            print(f"{NAMES[b]:7s} {LABELS[m]:15s}", end=" ", flush=True)
            cell = lactos_cell(b) if m == "lactos" else baseline_cell(b, m)
            if cell is None:
                print("(no maps)"); continue
            table[b][m] = cell
            print("AUROC %.4f  AP %.4f  AUPRO %.4f" % tuple(cell))
    os.makedirs(OUT, exist_ok=True)
    json.dump(table, open(os.path.join(OUT, "table1.json"), "w"), indent=1)

    # comparison with the paper
    rows, n_bad, n_cells = [], 0, 0
    for b in BENCHES:
        for m in methods:
            if m not in table[b]:
                continue
            for k, met in enumerate(("AUROC", "AP", "AUPRO")):
                ours, ref = table[b][m][k], EXPECTED[b][m][k]
                ok = round(ours, 4) == round(ref, 4)
                n_cells += 1; n_bad += (not ok)
                rows.append(f"| {NAMES[b]} | {LABELS[m]} | {met} | {ours:.4f} | {ref:.4f} | {ours - ref:+.4f} | {'ok' if ok else 'DIFF'} |")
    md = ["# Table 1 (reproduced) vs paper", "", "| Dataset | Method | Metric | reproduced | paper | delta | |", "|---|---|---|---:|---:|---:|---|"] + rows
    md += ["", f"{n_cells - n_bad}/{n_cells} cells agree at 4 decimals."]
    open(os.path.join(OUT, "table1.md"), "w").write("\n".join(md) + "\n")

    # LaTeX body in the paper's layout (methods as columns)
    tex = []
    for b in BENCHES:
        tex.append(r"      \multirow{3}{*}{%s}" % NAMES[b])
        for k, met in enumerate(("AUROC", "AP", "AUPRO")):
            vals = [table[b][m][k] if m in table[b] else float("nan") for m in METHODS]
            order = sorted(range(len(vals)), key=lambda j: -vals[j] if np.isfinite(vals[j]) else 1)
            cells = ["%.4f" % v if np.isfinite(v) else "--" for v in vals]
            if np.isfinite(vals[order[0]]): cells[order[0]] = r"\textbf{%s}" % cells[order[0]]
            if len(order) > 1 and np.isfinite(vals[order[1]]): cells[order[1]] = r"\underline{%s}" % cells[order[1]]
            tex.append("      & %-5s & " % met + " & ".join(cells) + r" \\")
        tex.append(r"      \midrule")
    open(os.path.join(OUT, "table1.tex"), "w").write("\n".join(tex) + "\n")

    print("\n" + "\n".join(md[2:]))
    print(f"\nwrote {OUT}/table1.{{json,md,tex}}")
    print("PASS" if n_bad == 0 and n_cells else f"FAIL: {n_bad} cell(s) differ from the paper")
    sys.exit(0 if n_bad == 0 and n_cells else 1)


if __name__ == "__main__":
    main()
