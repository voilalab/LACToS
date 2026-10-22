#!/usr/bin/env python
"""Stage 3: build Table 1 from the stage-2a (LACToS) and stage-2b (baseline) outputs and
compare every cell with the paper (expected/table1.json).

  python scripts/make_table1.py                   # all 7 methods x 3 benchmarks
  python scripts/make_table1.py --only lactos     # LACToS column only (after scripts/score_lactos.sh)
  python scripts/make_table1.py --only baselines
  python scripts/make_table1.py --maps reference  # the bundle's reference maps only (no GPU stage needed)

Which maps: by default (--maps auto) a cell is computed from your own maps under results/ (OUT);
when a method has no maps there and the bundle holds reference_results/ (the maintainers'
verification run, docs/REPRODUCIBILITY.md), the reference maps are used and the cell is marked
"ref". A benchmark that is only partly scored is reported as incomplete, never filled in.

Comparison: a cell is "ok" when |reproduced - paper| <= tolerance. Scoring on another GPU, or on
CPU, changes float32 round-off and moves a few values across histogram-bin edges, which shifts the
metrics in the third to fourth decimal (docs/REPRODUCIBILITY.md). DDAD's reverse process is
stochastic (eta = 1) and its seeded noise stream differs between torch builds and GPUs, so its
cells move more. Tolerances: 0.005, DDAD 0.03. --exact asks for agreement at 4 decimals, which
holds for the reference maps and for runs on the paper's hardware.

Writes results/table1.json, results/table1.md, results/table1.tex. Exit code 0 only when every
compared cell is within tolerance (or exact with --exact) and no benchmark is incomplete.

Metrics (full_metric_table.full_metrics): all valid pixels of all test images of a benchmark are
pooled; AUROC and AP (average precision) over that pool; AUPRO is the area under the
per-region-overlap curve up to FPR 0.3, normalised by 0.3.
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from full_metric_table import full_metrics            # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "baselines"))
from compose import pub_map                           # noqa: E402

OUT = os.environ.get("OUT", os.path.join(ROOT, "results"))
ASSETS = os.environ.get("ASSETS", os.path.join(ROOT, "assets"))
REFERENCE = os.path.join(ASSETS, "reference_results")   # lactos_maps/ and baseline_maps/ of the verification run
EXPECTED = json.load(open(os.path.join(ROOT, "expected", "table1.json")))
BENCHES = ["CT", "ffhq", "xray"]
NAMES = {"CT": "CHAOS", "ffhq": "FFHQ", "xray": "DVXray"}
METHODS = ["patchcore", "simplenet", "supersimplenet", "efficientad", "glass", "ddad", "lactos"]
LABELS = {"patchcore": "PatchCore", "simplenet": "SimpleNet", "supersimplenet": "SuperSimpleNet",
          "efficientad": "EfficientAD", "glass": "GLASS", "ddad": "DDAD", "lactos": "LACToS (Ours)"}
N_IMAGES = {"CT": 200, "ffhq": 100, "xray": 144}
TOLERANCE = {"ddad": 0.03}; TOLERANCE_DEFAULT = 0.005

# LACToS operating points (paper Table 5): dump tags produced by scripts/score_lactos.sh
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


def _load_dumps(root, bench, tag, sets):
    """{'<set>/<file>': npz} for every dump of the tag, keyed so that two tags can be paired by image."""
    return {f"{s}/{os.path.basename(f)}": np.load(f)
            for s in sets for f in sorted(glob.glob(os.path.join(root, "lactos_maps", bench, tag, s, "samples_*.npz")))}


def _valid(z):
    v = z["valid_mask"]
    return v.astype(bool) if v.shape == z["delta_map"].shape else np.ones(z["delta_map"].shape, bool)


def lactos_cell(bench, root):
    """Pooled AUROC / AP / AUPRO of LACToS at the paper's operating point, from the dumps under <root>/lactos_maps.
    score = d_rgb + lam * (d_full - d_rgb): lam=1 -> full dump only, lam=0 -> RGB-only dump only.
    Returns None (no maps), "incomplete", or [auroc, ap, aupro]."""
    op = LACTOS[bench]
    full = _load_dumps(root, bench, op["full"], op["sets"]) if op["lam"] > 0 else None
    rgb = _load_dumps(root, bench, op["rgb"], op["sets"]) if op["lam"] < 1 else None
    base = full if full is not None else rgb
    if not base and not (full or rgb):
        return None
    if full is not None and rgb is not None and set(full) != set(rgb):
        print(f"  [incomplete] {bench}: the full and the RGB-only runs cover different images "
              f"({len(full)} vs {len(rgb)}); re-run scripts/score_lactos.sh {bench}")
        return "incomplete"
    if len(base) != N_IMAGES[bench]:
        print(f"  [incomplete] {bench}: {len(base)} LACToS dumps, expected {N_IMAGES[bench]} (run scripts/score_lactos.sh {bench})")
        return "incomplete"
    maps, gts, valids = [], [], []
    for key, z in base.items():
        d = np.nan_to_num(z["delta_map"].astype(np.float64))
        if full is not None and rgb is not None:
            r = np.nan_to_num(rgb[key]["delta_map"].astype(np.float64))
            d = r + op["lam"] * (d - r)
        d = d.astype(np.float32)
        maps.append(gaussian_filter(d, op["sigma"], mode="nearest") if op["sigma"] > 0 else d)
        gts.append((z["gt_mask"] > 0).astype(np.uint8)); valids.append(_valid(z))
    r = full_metrics(maps, gts, valids)
    return [r["pooled_auroc"], r["pooled_aupr"], r["aupro"]]


def baseline_cell(bench, method, root):
    """Same for a baseline, from <root>/baseline_maps/<bench>/<method>/<setting>/*.npz."""
    setting, knob = BASELINE[method][bench]
    bench_dir = bench.lower()                       # baselines/table1_common.py names the benchmarks ffhq / ct / xray
    files = sorted(glob.glob(os.path.join(root, "baseline_maps", bench_dir, method, setting, "*.npz")))
    if not files:
        return None
    if len(files) != N_IMAGES[bench]:
        print(f"  [incomplete] {bench}/{method}: {len(files)} maps, expected {N_IMAGES[bench]} (run scripts/run_baselines.sh {bench_dir})")
        return "incomplete"
    maps, gts, valids = [], [], []
    for f in files:
        z = dict(np.load(f))
        maps.append(pub_map(method, bench_dir, z, knob))
        gts.append(z["gt_mask"].astype(np.uint8)); valids.append(z["valid_mask"].astype(bool))
    r = full_metrics(maps, gts, valids)
    return [r["pooled_auroc"], r["pooled_aupr"], r["aupro"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["lactos", "baselines", "all"], default="all")
    ap.add_argument("--maps", choices=["auto", "mine", "reference"], default="auto",
                    help="auto: your maps under OUT, the bundle's reference_results/ for methods you did not run; "
                         "mine: never fall back; reference: the reference maps only")
    ap.add_argument("--exact", action="store_true", help="require agreement at 4 decimals instead of the tolerance")
    a = ap.parse_args()
    methods = [m for m in METHODS if a.only == "all" or (m == "lactos") == (a.only == "lactos")]
    have_ref = os.path.isdir(REFERENCE)
    if a.maps == "reference" and not have_ref:
        sys.exit(f"no reference maps at {REFERENCE} (the bundle's reference_results/ folder)")
    roots = {"auto": [("run", OUT)] + ([("ref", REFERENCE)] if have_ref else []),
             "mine": [("run", OUT)], "reference": [("ref", REFERENCE)]}[a.maps]
    table = {b: {} for b in BENCHES}
    source = {b: {} for b in BENCHES}
    incomplete = []
    for b in BENCHES:
        for m in methods:
            print(f"{NAMES[b]:7s} {LABELS[m]:15s}", end=" ", flush=True)
            cell, src = None, None
            for src, root in roots:                 # first root that has maps decides
                cell = lactos_cell(b, root) if m == "lactos" else baseline_cell(b, m, root)
                if cell is not None:
                    break
            if cell is None:
                print("(no maps)"); continue
            if cell == "incomplete":
                incomplete.append(f"{NAMES[b]}/{LABELS[m]}"); continue
            table[b][m] = cell; source[b][m] = src
            print("AUROC %.4f  AP %.4f  AUPRO %.4f" % tuple(cell) + ("   [ref]" if src == "ref" else ""))
    os.makedirs(OUT, exist_ok=True)
    json.dump({"table": table, "maps": source}, open(os.path.join(OUT, "table1.json"), "w"), indent=1)

    # comparison with the paper
    rows, n_bad, n_exact, n_cells = [], 0, 0, 0
    for b in BENCHES:
        for m in methods:
            if m not in table[b]:
                continue
            tol = 0 if a.exact else TOLERANCE.get(m, TOLERANCE_DEFAULT)
            for k, met in enumerate(("AUROC", "AP", "AUPRO")):
                ours, ref = table[b][m][k], EXPECTED[b][m][k]
                exact = round(ours, 4) == round(ref, 4)
                ok = exact or abs(ours - ref) <= tol
                n_cells += 1; n_bad += (not ok); n_exact += exact
                rows.append(f"| {NAMES[b]} | {LABELS[m]} | {met} | {ours:.4f} | {ref:.4f} | {ours - ref:+.4f} | "
                            f"{'ok' if ok else 'DIFF'} | {source[b][m]} |")
    md = ["# Table 1 (reproduced) vs paper", "",
          "| Dataset | Method | Metric | reproduced | paper | delta | | maps |", "|---|---|---|---:|---:|---:|---|---|"] + rows
    crit = "at 4 decimals" if a.exact else f"within tolerance (|delta| <= {TOLERANCE_DEFAULT}, DDAD {TOLERANCE['ddad']})"
    md += ["", f"{n_cells - n_bad}/{n_cells} cells {crit}" + (f"; {n_exact}/{n_cells} agree at 4 decimals." if not a.exact else ".")]
    if any(source[b][m] == "ref" for b in BENCHES for m in source[b]):
        md += ["", "maps = ref: computed from the bundle's reference_results/ (the maintainers' run), not from your run."]
    if incomplete:
        md += ["", "Incomplete (not compared): " + ", ".join(incomplete)]
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
    ok = n_bad == 0 and n_cells > 0 and not incomplete
    if ok:
        print(f"PASS: every cell {crit}")
    elif n_cells == 0:
        print("FAIL: no complete benchmark to compare" + (f"; incomplete: {', '.join(incomplete)}" if incomplete else ""))
    else:
        print(f"FAIL: {n_bad} cell(s) outside the tolerance" + (f"; incomplete: {', '.join(incomplete)}" if incomplete else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
