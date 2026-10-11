"""Turn the saved raw quantities of a baseline into its published anomaly map.

compose(method, z, knob)      unsmoothed 256x256 map for one post-hoc knob value
pub_map(method, bench, z, knob)  the map at the published blur composition: each method's
                               internal sigma=4 blur at its input resolution (none for
                               EfficientAD), resize to 256, then the evaluation sigma=5.
Knobs: PatchCore k (Table 1: 1), EfficientAD alpha = weight of the student-teacher map
(Table 1: CT 0, FFHQ 0.7, X-ray 0.2), DDAD v = pixel/feature weight (Table 1: CT 1,
FFHQ 4, X-ray 4); None for SimpleNet / SuperSimpleNet / GLASS.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

import table1_common as C


def _bilinear(m, size):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def compose(method, z, knob):
    if method == "patchcore":
        d = z["nn_dist"][..., :knob].mean(-1)                           # upstream: mean of the k NN distances
        return _bilinear(_bilinear(d, int(z["input_size"])), C.SIZE)   # patch grid -> input size -> 256 (two steps, as upstream)
    if method == "efficientad":
        return knob * z["map_st"] + (1.0 - knob) * z["map_ae"]
    if method == "ddad":
        f, i = z["f_d"], z["i_d"]
        return f + knob * (f.max() / i.max()) * i                       # DDAD/anomaly_map.py without the blur
    return z["map"]


def pub_map(method, bench, z, knob):
    if method == "patchcore":
        d = z["nn_dist"][..., :knob].mean(-1); S = int(z["input_size"])
        m = _bilinear(gaussian_filter(_bilinear(d, S), 4), C.SIZE)
    elif method in ("simplenet", "glass"):
        S = C.input_size(method, bench)       # the map was saved at 256; the upstream blur acts at input resolution
        m = _bilinear(gaussian_filter(_bilinear(z["map"], S), 4), C.SIZE) if S != 256 else gaussian_filter(z["map"], 4)
    elif method in ("supersimplenet", "ddad"):
        m = gaussian_filter(compose(method, z, knob), 4)
    else:
        m = compose(method, z, knob)          # EfficientAD has no internal blur
    return gaussian_filter(m.astype(np.float32), 5, mode="nearest")
