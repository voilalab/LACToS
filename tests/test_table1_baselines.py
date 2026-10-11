"""Regression test: the shipped baseline inference reproduces the lab's raw maps behind Table 1.

Runs `scripts/30_run_baselines.sh <bench>` with LIMIT=2 (first two images of every set) and
compares each method's published-blur map (baselines/compose.pub_map at the Table 1 knob)
with the lab's saved raw maps (results_eval/baseline_sweep/maps). Feature methods must
correlate at r >= 0.999 (same weights, same preprocessing); DDAD uses the same per-image seed
and must agree to 1e-4.

Needs: a GPU, the four conda envs, third_party/ clones, the bundle and the lab tree. Skipped otherwise.
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(os.environ.get("ASSETS", ROOT / "assets"))
LAB = Path(os.environ.get("LAB_REPO", "/usr/scratch/jhong392/workspace/Statistical_OOD_detection")) / "results_eval/baseline_sweep/maps"
sys.path.insert(0, str(ROOT / "baselines"))

PICKS = {  # method -> bench -> (setting, knob)
    "patchcore": {b: ("default", 1) for b in ("ffhq", "ct", "xray")},
    "simplenet": {b: ("default", None) for b in ("ffhq", "ct", "xray")},
    "supersimplenet": {b: ("default", None) for b in ("ffhq", "ct", "xray")},
    "glass": {b: ("default", None) for b in ("ffhq", "ct", "xray")},
    "efficientad": {"ct": ("nopad", 0.0), "ffhq": ("default", 0.7), "xray": ("nopad", 0.2)},
    "ddad": {"ct": ("w2_T250_s25_e1", 1.0), "ffhq": ("w2_T250_s25_e1", 4.0), "xray": ("w4_T250_s25_e1", 4.0)},
}

pytestmark = [
    pytest.mark.requires_gpu, pytest.mark.slow,
    pytest.mark.skipif(not (ASSETS / "checkpoints").is_dir(), reason="asset bundle not extracted"),
    pytest.mark.skipif(not LAB.is_dir(), reason="lab reference maps not reachable"),
    pytest.mark.skipif(not (ROOT / "third_party" / "GLASS").is_dir(), reason="third_party clones missing"),
]


def _lab_name(fname):
    """our ctn ids are image_6000+i; the lab used image_%04d."""
    s, sid = fname[:-4].split("__", 1)
    if s == "ctn":
        sid = "image_%04d" % (int(sid.split("_")[1]) - 6000)
    return f"{s}__{sid}.npz"


@pytest.mark.parametrize("bench", ["ffhq", "ct", "xray"])
def test_baselines_reproduce_lab_maps(bench, tmp_path):
    env = dict(os.environ, ASSETS=str(ASSETS), OUT=str(tmp_path), LIMIT="2", GPU=os.environ.get("GPU", "0"))
    r = subprocess.run(["bash", str(ROOT / "scripts" / "30_run_baselines.sh"), bench], cwd=ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + "".join(open(f).read()[-1500:] for f in glob.glob(str(tmp_path / "logs" / "*.log")))
    from compose import pub_map
    report = []
    for method, picks in PICKS.items():
        setting, knob = picks[bench]
        ours = sorted(glob.glob(str(tmp_path / "baseline_maps" / bench / method / setting / "*.npz")))
        assert len(ours) >= 2, f"{bench}/{method}: no maps written"
        for f in ours:
            ref = LAB / bench / method / setting / _lab_name(os.path.basename(f))
            assert ref.exists(), f"no lab reference {ref}"
            a, b = dict(np.load(f)), dict(np.load(ref))
            assert np.array_equal(a["gt_mask"] > 0, b["gt_mask"] > 0), f"{bench}/{method}/{f}: gt differs"
            assert np.array_equal(a["valid_mask"] > 0, b["valid_mask"] > 0), f"{bench}/{method}/{f}: valid mask differs"
            ma, mb = pub_map(method, bench, a, knob), pub_map(method, bench, b, knob)
            rr = np.corrcoef(ma.ravel(), mb.ravel())[0, 1]
            report.append(f"{bench}/{method}/{os.path.basename(f)}: r={rr:.6f} max|d|={np.abs(ma - mb).max():.3g}")
            assert rr >= 0.999, report[-1]
            if method == "ddad":
                assert np.allclose(ma, mb, atol=1e-4), report[-1]
    print("\n".join(report))
