"""Generate the toy datasets of the paper (Section 3, Appendix B and C).

Every image holds four objects, one near the centre of each quadrant (origin at the image
centre, y down); the position is Gaussian (sigma 8 px) around the anchor, the grey level is
uniform, and every class has the area of a radius-20 circle:

    Q1 (192, 64)   inlier: CIRCLE ONLY      anomalous test images: a square
    Q2  (64, 64)   drawn from the dataset's law
    Q3  (64, 192)  idem
    Q4 (192, 192)  idem

Toy experiment 1 (`quad_L1` .. `quad_L4`): Q2-Q4 draw uniformly from a vocabulary that
grows by one class per level, L1 {circle, square}, L2 + triangle, L3 + ellipse, L4 + star.
Test set `q1_square`: 50 images with a square at Q1.

Toy experiment 2 (`quadp_p50` .. `quadp_p10`): the L1 vocabulary with P(square) = p at
Q2-Q4 (0.5, 0.4, 0.3, 0.2, 0.1); the datasets share one random stream, so they differ
only in which objects are squares. `quadp_p50` is `quad_L1` (same law; its train/good
and test/good are symlinks and its prior is quad_L1's). Test set `q1_square_plus`: the
same 50 images for every p, each with a square at Q1 and at least one square at Q2-Q4.

Layout (MVTec-style, what toy/train_prior.py and the scoring scripts read):
    <data_root>/<name>/train/good/*.png, test/good/*.png, test/<variant>/*.png,
    ground_truth/<variant>/*_mask.png, meta.json

    python toy/gen_data.py                      # everything: 4 levels + 5 p values (~10 min, 8 processes)
    python toy/gen_data.py --levels 1 2         # a subset
    python toy/gen_data.py --p_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repository root, for `toy.*`
from toy.shapes import (Circle, Ellipse, RegularPolygon, Square, Star, circumradius_for_area,
                        circumradius_for_star_area, render)

TOY_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TOY_DIR / "config.yaml"
DEFAULT_DATA_ROOT = TOY_DIR / "data"

RADIUS = 20.0
TARGET_AREA = float(np.pi * RADIUS ** 2)          # 1256.637
SIGMA_POS = 8.0
ASPECT = 2.5                                      # inlier ellipse, equal-area
STAR_RATIO = 0.382

# Image coords, y-down. Index 0 is Q1 (+x, +y in maths convention) = top-right.
ANCHORS: List[Tuple[float, float]] = [
    (192.0, 64.0),    # Q1
    (64.0, 64.0),     # Q2
    (64.0, 192.0),    # Q3
    (192.0, 192.0),   # Q4
]
Q1 = 0

LEVELS: Dict[int, List[str]] = {
    1: ["circle", "square"],
    2: ["circle", "square", "triangle"],
    3: ["circle", "square", "triangle", "ellipse"],
    4: ["circle", "square", "triangle", "ellipse", "star"],
}
OOD_KIND = "square"                               # fixed across every level
LEVEL_VARIANT = "q1_square"                       # test subfolder of toy 1

# Disjoint seed ranges per level and split.
SEED_BASE = {1: 71_000_000, 2: 72_000_000, 3: 73_000_000, 4: 74_000_000}
OFF_TEST_GOOD, OFF_OOD = 500_000, 900_000

# p-sweep (toy 2): one seed stream for every p; each object consumes the same five random
# numbers whatever its class, so the squares at a lower p are a subset of those at a higher p.
P_SWEEP = {"ps": [0.4, 0.3, 0.2, 0.1], "test_seed": 89_000_000, "n_test": 50,
           "test_variant": "q1_square_plus"}
SEED_BASE_P = 80_000_000

_POLY_SIDES = {"square": 4, "triangle": 3}
_ELLIPSE_A = float(np.sqrt(TARGET_AREA * ASPECT / np.pi))      # 31.623
_ELLIPSE_B = _ELLIPSE_A / ASPECT                               # 12.649
_STAR_R = circumradius_for_star_area(TARGET_AREA, STAR_RATIO)  # 33.456


def load_config(path=DEFAULT_CONFIG) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_rgb(gray: np.ndarray, path: Path) -> None:
    """3-channel PNG (R = G = B): every downstream loader expects 3 channels."""
    Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB").save(path)


def margin_for(kind: str) -> float:
    """Largest centre-to-boundary distance: the border inset that keeps the whole shape in frame."""
    if kind == "circle":
        return RADIUS
    if kind == "ellipse":
        return _ELLIPSE_A
    if kind == "star":
        return _STAR_R
    return circumradius_for_area(TARGET_AREA, _POLY_SIDES[kind])


def make_shape(kind: str, rng: np.random.Generator, cx: float, cy: float,
               intensity: int, theta_u: Optional[float] = None):
    """One equal-area shape of class `kind`, rotated uniformly over its symmetry period.
    `theta_u` in [0, 1) supplies the rotation draw instead of `rng` (the p-sweep)."""
    rot = (lambda period: float(rng.uniform(0.0, period))) if theta_u is None \
        else (lambda period: float(theta_u) * period)
    if kind == "circle":
        return Circle(cx, cy, RADIUS, intensity)
    if kind == "ellipse":
        return Ellipse(cx, cy, _ELLIPSE_A, _ELLIPSE_B, rot(np.pi), intensity)
    if kind == "star":
        return Star(cx, cy, _STAR_R, STAR_RATIO, rot(2.0 * np.pi / 5), intensity)
    n = _POLY_SIDES[kind]
    R = circumradius_for_area(TARGET_AREA, n)
    if kind == "square":
        return Square(cx, cy, R * float(np.sqrt(2.0)), rot(np.pi / 2), intensity)
    return RegularPolygon(cx, cy, R, n, rot(2.0 * np.pi / n), intensity)


def _intensity(rng: np.random.Generator, cfg: Dict) -> int:
    lo, hi = int(cfg["inlier"]["intensity_min"]), int(cfg["inlier"]["intensity_max"])
    return int(rng.integers(lo, hi + 1))


def _jitter(rng: np.random.Generator, anchor: Tuple[float, float], margin: float, size: int) -> Tuple[float, float]:
    """Gaussian around the anchor, clipped so the shape stays whole in frame."""
    ax, ay = anchor
    cx = float(np.clip(rng.normal(ax, SIGMA_POS), margin, size - margin))
    cy = float(np.clip(rng.normal(ay, SIGMA_POS), margin, size - margin))
    return cx, cy


def sample_objects(rng: np.random.Generator, cfg: Dict, classes: List[str], q1_kind: str = "circle") -> List:
    """One object per anchor, in anchor order. Q1 is `q1_kind`; Q2-Q4 draw uniformly from `classes`."""
    size = int(cfg["image_size"])
    out = []
    for i, anchor in enumerate(ANCHORS):
        kind = q1_kind if i == Q1 else classes[int(rng.integers(0, len(classes)))]
        cx, cy = _jitter(rng, anchor, margin_for(kind), size)
        out.append(make_shape(kind, rng, cx, cy, _intensity(rng, cfg)))
    return out


def _meta(objs) -> List[Dict]:
    out = []
    for o in objs:
        d = {"kind": type(o).__name__}
        d.update({f: getattr(o, f) for f in o.__dataclass_fields__})
        out.append(d)
    return out


def make_inlier(seed: int, cfg: Dict, classes: List[str]):
    rng = np.random.default_rng(seed)
    objs = sample_objects(rng, cfg, classes)
    img = render(objs, int(cfg["image_size"]), int(cfg["supersample"]))
    return img, {"seed": seed, "objects": _meta(objs)}


def make_ood(seed: int, cfg: Dict, classes: List[str]):
    """Same law, except Q1 hosts a square. Drawn LAST so it is never occluded and the ground
    truth is exactly its rendered footprint."""
    rng = np.random.default_rng(seed)
    size, ss = int(cfg["image_size"]), int(cfg["supersample"])
    objs = sample_objects(rng, cfg, classes, q1_kind=OOD_KIND)
    anom = objs[Q1]
    others = [o for i, o in enumerate(objs) if i != Q1]
    img = render(others + [anom], size, ss)
    mask = ((anom.coverage(size, ss) >= 0.5) * 255).astype(np.uint8)
    return img, mask, {"seed": seed, "objects": _meta(others), "anomaly": _meta([anom])[0]}


# ── p-sweep ───────────────────────────────────────────────────────────────────────────────

def p_name(p: float) -> str:
    return f"quadp_p{int(round(p * 100)):02d}"


def sample_objects_p(rng: np.random.Generator, cfg: Dict, p: float, q1_kind: str = "circle") -> List:
    """One object per anchor; Q2-Q4 are a square with probability `p`, else a circle. Every
    object consumes the same five draws (class, dx, dy, intensity, rotation) whatever its class."""
    size = int(cfg["image_size"])
    out = []
    for i, (ax, ay) in enumerate(ANCHORS):
        u_class, dx, dy = rng.random(), rng.normal(), rng.normal()
        inten, theta_u = _intensity(rng, cfg), rng.random()
        kind = q1_kind if i == Q1 else ("square" if u_class >= 1.0 - p else "circle")
        m = margin_for(kind)
        cx = float(np.clip(ax + SIGMA_POS * dx, m, size - m))
        cy = float(np.clip(ay + SIGMA_POS * dy, m, size - m))
        out.append(make_shape(kind, rng, cx, cy, inten, theta_u=theta_u))
    return out


def make_inlier_p(seed: int, cfg: Dict, p: float):
    rng = np.random.default_rng(seed)
    objs = sample_objects_p(rng, cfg, p)
    img = render(objs, int(cfg["image_size"]), int(cfg["supersample"]))
    return img, {"seed": seed, "p_square": p, "objects": _meta(objs)}


def make_ood_plus(seed: int, cfg: Dict):
    """A square at Q1 AND at least one square among Q2-Q4 (drawn at p = 0.5, redrawn until one
    is a square). The same images serve every p, so a change across the sweep is the prior alone."""
    rng = np.random.default_rng(seed)
    size, ss = int(cfg["image_size"]), int(cfg["supersample"])
    while True:
        objs = sample_objects_p(rng, cfg, 0.5, q1_kind="square")
        if any(type(o).__name__ == "Square" for o in objs[1:]):
            break
    anom, others = objs[Q1], objs[1:]
    img = render(others + [anom], size, ss)
    mask = ((anom.coverage(size, ss) >= 0.5) * 255).astype(np.uint8)
    return img, mask, {"seed": seed, "objects": _meta(others), "anomaly": _meta([anom])[0]}


# ── writers ───────────────────────────────────────────────────────────────────────────────

def _write_shared_test(cat: Path, cfg: Dict) -> None:
    v = P_SWEEP["test_variant"]
    (cat / f"test/{v}").mkdir(parents=True, exist_ok=True)
    (cat / f"ground_truth/{v}").mkdir(parents=True, exist_ok=True)
    for i in range(P_SWEEP["n_test"]):
        img, mask, _ = make_ood_plus(P_SWEEP["test_seed"] + i, cfg)
        _save_rgb(img, cat / f"test/{v}/{i:03d}.png")
        Image.fromarray(mask, mode="L").save(cat / f"ground_truth/{v}/{i:03d}_mask.png")


def write_level(level: int, cfg: Dict, data_root: Path) -> Dict:
    classes = LEVELS[level]
    name = f"quad_L{level}"
    cat = data_root / name
    train_good, test_good = cat / "train/good", cat / "test/good"
    test_ood, gt_ood = cat / f"test/{LEVEL_VARIANT}", cat / f"ground_truth/{LEVEL_VARIANT}"
    for d in (train_good, test_good, test_ood, gt_ood):
        d.mkdir(parents=True, exist_ok=True)
    base = SEED_BASE[level]
    n_tr, n_tg = int(cfg["counts"]["train_good"]), int(cfg["counts"]["test_good"])
    n_to = int(cfg["counts"]["test_anomalous"])
    fills, class_counts = [], {}
    for i in range(n_tr):
        img, m = make_inlier(base + i, cfg, classes)
        _save_rgb(img, train_good / f"{i:05d}.png")
        if i < 200:
            fills.append(float((img > 0).mean()))
        for o in m["objects"]:
            class_counts[o["kind"]] = class_counts.get(o["kind"], 0) + 1
    for i in range(n_tg):
        img, _ = make_inlier(base + OFF_TEST_GOOD + i, cfg, classes)
        _save_rgb(img, test_good / f"{i:03d}.png")
    areas = []
    for i in range(n_to):
        img, mask, _ = make_ood(base + OFF_OOD + i, cfg, classes)
        _save_rgb(img, test_ood / f"{i:03d}.png")
        Image.fromarray(mask, mode="L").save(gt_ood / f"{i:03d}_mask.png")
        areas.append(float((mask > 0).mean()))
    meta = {"level": level, "classes": classes, "anchors": ANCHORS, "q1_index": Q1, "sigma_pos": SIGMA_POS,
            "target_area": TARGET_AREA, "ood_kind": OOD_KIND, "train_class_counts": class_counts,
            "mean_fill": float(np.mean(fills)), "mean_anomalous_fraction": float(np.mean(areas)), "config": cfg}
    (cat / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name} {classes}: {n_tr}/{n_tg}/{n_to} images, fill {np.mean(fills) * 100:.1f}%, "
          f"anomaly {np.mean(areas) * 100:.2f}% -> {cat}", flush=True)
    return meta


def write_p(p: float, cfg: Dict, data_root: Path) -> Dict:
    name = p_name(p)
    cat = data_root / name
    tr, tg = cat / "train/good", cat / "test/good"
    for d in (tr, tg):
        d.mkdir(parents=True, exist_ok=True)
    n_tr, n_tg = int(cfg["counts"]["train_good"]), int(cfg["counts"]["test_good"])
    squares = total = 0
    for i in range(n_tr):
        img, m = make_inlier_p(SEED_BASE_P + i, cfg, p)
        _save_rgb(img, tr / f"{i:05d}.png")
        kinds = [o["kind"] for o in m["objects"][1:]]
        squares += kinds.count("Square"); total += len(kinds)
    for i in range(n_tg):
        _save_rgb(make_inlier_p(SEED_BASE_P + OFF_TEST_GOOD + i, cfg, p)[0], tg / f"{i:03d}.png")
    _write_shared_test(cat, cfg)
    meta = {"p_square_q2_q4": p, "measured_square_fraction_q2_q4": squares / max(total, 1), "q1": "circle only",
            "classes": ["circle", "square"], "anchors": ANCHORS, "sigma_pos": SIGMA_POS,
            "test_variant": P_SWEEP["test_variant"],
            "test": "square at Q1 and >= 1 square at Q2-Q4; identical images for every p",
            "seed_base": SEED_BASE_P, "test_seed": P_SWEEP["test_seed"], "config": cfg}
    (cat / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: {n_tr}/{n_tg} images, square fraction at Q2-Q4 {meta['measured_square_fraction_q2_q4']:.3f} "
          f"(target {p}) -> {cat}", flush=True)
    return meta


def write_p50_reference(cfg: Dict, data_root: Path) -> None:
    """p = 0.5 is quad_L1's law: link its images (and use its prior), add the shared test set."""
    name, src = p_name(0.5), data_root / "quad_L1"
    cat = data_root / name
    for sub in ("train/good", "test/good"):
        link = cat / sub
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(os.path.relpath(src / sub, link.parent), target_is_directory=True)
    _write_shared_test(cat, cfg)
    (cat / "meta.json").write_text(json.dumps({
        "p_square_q2_q4": 0.5, "reuses": "quad_L1 train/good, test/good and prior",
        "test_variant": P_SWEEP["test_variant"], "test_seed": P_SWEEP["test_seed"]}, indent=2))
    print(f"{name}: links to quad_L1 + shared test set -> {cat}", flush=True)


def _job(args):
    kind, value, cfg, data_root = args
    return write_level(value, cfg, data_root) if kind == "level" else write_p(value, cfg, data_root)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", type=int, nargs="*", choices=sorted(LEVELS), default=None,
                   help="toy 1 levels to generate (default with no flags: all four)")
    p.add_argument("--p_sweep", action="store_true", help="toy 2: quadp_p50 (links quad_L1) and quadp_p40..p10")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    cfg = load_config(args.config)
    data_root = Path(args.data_root)
    everything = args.levels is None and not args.p_sweep
    levels = sorted(LEVELS) if everything or args.levels == [] else (args.levels or [])
    do_p = everything or args.p_sweep
    jobs = [("level", L, cfg, data_root) for L in levels]
    if do_p:
        jobs += [("p", pv, cfg, data_root) for pv in P_SWEEP["ps"]]
    with Pool(min(args.workers, max(1, len(jobs)))) as pool:
        pool.map(_job, jobs)
    if do_p:
        if not (data_root / "quad_L1" / "train/good").is_dir():
            raise SystemExit("quadp_p50 links quad_L1: generate level 1 first (python toy/gen_data.py --levels 1)")
        write_p50_reference(cfg, data_root)


if __name__ == "__main__":
    main()
