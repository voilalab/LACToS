"""Shared definitions for the six baselines of Table 1 (data layout, checkpoints, preprocessing).

Not named `common.py` on purpose: SimpleNet, GLASS and SuperSimpleNet each import their own
top-level `common` module, and `baselines/` is on sys.path when they are loaded.

Benchmarks (every image is scored at 256x256; all valid pixels are pooled):
  ffhq : 22 faces + 28 moreface (anomalous) + 50 faces_normal                         = 100
  ct   : 100 anomalous (cta) + 100 normal same-base slices (ctn), body-mask valid region = 200
  xray : 72 scissors (xrsc) + 72 good (xrgd)                                          = 144
Every inference script writes one npz per image: <out_root>/<bench>/<method>/<setting>/<set>__<id>.npz
holding the UNSMOOTHED quantities that compose.py turns into the published map.
Python 3.8 compatible (patchcore / simplenet envs).
"""
import os, zlib
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.environ.get("ASSETS", os.path.join(ROOT, "assets"))
OUT_ROOT = os.path.join(os.environ.get("OUT", os.path.join(ROOT, "results")), "baseline_maps")
THIRD = os.path.join(ROOT, "third_party")
UPSTREAM = {"patchcore": os.path.join(THIRD, "patchcore-inspection", "src"),
            "simplenet": os.path.join(THIRD, "SimpleNet"),
            "glass": os.path.join(THIRD, "GLASS"),
            "supersimplenet": os.path.join(THIRD, "SuperSimpleNet")}
SIZE = 256
BENCHES = ("ffhq", "ct", "xray")
FEATURE_METHODS = ("patchcore", "simplenet", "glass", "supersimplenet", "efficientad")

# ---------------------------------------------------------------- checkpoints
_CK = os.path.join(ASSETS, "checkpoints", "baselines")
CKPT = {
    "patchcore":      {b: os.path.join(_CK, b, "patchcore") for b in BENCHES},            # faiss index + params.pkl
    "simplenet":      {b: os.path.join(_CK, b, "simplenet", "ckpt.pth") for b in BENCHES},
    "glass":          {b: os.path.join(_CK, b, "glass", "ckpt_best.pth") for b in BENCHES},
    "supersimplenet": {b: os.path.join(_CK, b, "supersimplenet", "weights.pt") for b in BENCHES},
    "efficientad":    {b: os.path.join(_CK, b, "efficientad") for b in BENCHES},          # {teacher,student,autoencoder}_final.pth + norm_stats.pt
    "ddad":           {b: os.path.join(_CK, b, "ddad", "unet.pth") for b in BENCHES},
}
# DDAD feature extractor used for the published numbers (frozen ImageNet weights)
DDAD_FE = {"ffhq": "wide_resnet101_2", "ct": "resnet101", "xray": "resnet101"}

# Per-(method, bench) input preprocessing of the runs behind the published numbers: (resize, centre_crop or None)
PREP = {
    "patchcore":      {"ffhq": (256, 224), "ct": ((224, 224), None), "xray": ((224, 224), None)},
    "simplenet":      {"ffhq": (329, 288), "ct": ((256, 256), None), "xray": ((288, 288), None)},
    "glass":          {"ffhq": (288, 288), "ct": ((256, 256), None), "xray": ((288, 288), None)},
    "supersimplenet": {"ffhq": ((256, 256), None), "ct": ((256, 256), None), "xray": ((256, 256), None)},
    "efficientad":    {"ffhq": ((256, 256), None), "ct": ((256, 256), None), "xray": ((256, 256), None)},
}


def input_size(method, bench):
    r, c = PREP[method][bench]
    return c if c else r[0]


# ---------------------------------------------------------------- data
# (set, image dir, GT dir, valid-mask dir or None)
def _ffhq_sets():
    D = os.path.join(ASSETS, "data", "ffhq")
    return [(s, os.path.join(D, s, "test"), os.path.join(D, s, "ground_truth"), None)
            for s in ("faces", "moreface", "faces_normal")]


def _ct_sets():
    D = os.path.join(ASSETS, "data", "CT")
    return [("cta", os.path.join(D, "test", "cta"), os.path.join(D, "ground_truth", "cta"), os.path.join(D, "masks", "masks_body_ood_default_large")),
            # normal slices carry ids image_6000+i; the baselines were evaluated inside the handoff body masks
            ("ctn", os.path.join(D, "test", "ctn"), os.path.join(D, "ground_truth", "ctn"), os.path.join(D, "body_masks_baselines"))]


def _xray_sets():
    D = os.path.join(ASSETS, "data", "xray")
    return [(s, os.path.join(D, "test", s), os.path.join(D, "ground_truth", s), None) for s in ("xrsc", "xrgd")]


SETS = {"ffhq": _ffhq_sets, "ct": _ct_sets, "xray": _xray_sets}
N_IMAGES = {"ffhq": 100, "ct": 200, "xray": 144}


def _key(stem):
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def list_images(bench):
    """[(set, id, img_path, gt_path, valid_path_or_None)] in a fixed order."""
    out = []
    for s, img_dir, gt_dir, valid_dir in SETS[bench]():
        stems = sorted((f[:-4] for f in os.listdir(img_dir) if f.endswith(".png")), key=_key)
        for sid in stems:
            out.append((s, sid, os.path.join(img_dir, sid + ".png"), os.path.join(gt_dir, sid + "_mask.png"),
                        os.path.join(valid_dir, sid + "_mask.png") if valid_dir else None))
    assert len(out) == N_IMAGES[bench], (bench, len(out))
    return out


def load_gt(gt_path):
    """Binary GT at 256x256. CT lesion masks are 512x512 and are subsampled [::2, ::2] (as the lab
    scripts did); other non-256 masks (1254px faces) use nearest-neighbour resizing."""
    g = np.array(Image.open(gt_path).convert("L"))
    if g.shape == (512, 512):
        g = g[::2, ::2]
    elif g.shape != (SIZE, SIZE):
        g = np.array(Image.fromarray(g).resize((SIZE, SIZE), Image.NEAREST))
    return (g > 0).astype(np.uint8)


def load_valid(valid_path):
    if valid_path is None:
        return np.ones((SIZE, SIZE), np.uint8)
    b = np.array(Image.open(valid_path).convert("L")) > 0
    if b.shape != (SIZE, SIZE):
        b = np.array(Image.fromarray(b.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.NEAREST)) > 0
    return b.astype(np.uint8)


def legacy_id(set_name, sid):
    """Image id as used in the lab runs (CT normal slices were image_0000..0099 there)."""
    if set_name == "ctn" and sid.startswith("image_6"):
        return "image_%04d" % (int(sid.split("_")[1]) - 6000)
    return sid


def image_seed(set_name, sid):
    """Per-image noise seed of the DDAD baseline (crc32 of 'set/id' as in the lab runs)."""
    return zlib.crc32(("%s/%s" % (set_name, legacy_id(set_name, sid))).encode()) & 0x7FFFFFFF


def out_dir(root, bench, method, setting):
    return os.path.join(root, bench, method, setting)


def npz_name(set_name, sid):
    return "%s__%s.npz" % (set_name, sid)


def n_done(d):
    return sum(1 for f in os.listdir(d) if f.endswith(".npz")) if os.path.isdir(d) else 0


# ---------------------------------------------------------------- fixed inference settings
PATCHCORE_KMAX = 15     # neighbours saved; Table 1 uses k = 1


def ddad_setting(w, T, skip, eta):
    return "w%g_T%d_s%d_e%g" % (w, T, skip, eta)
