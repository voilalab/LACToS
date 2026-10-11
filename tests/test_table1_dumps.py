"""Regression test: the shipped scorer reproduces the delta-map dumps behind Table 1.

Runs `scripts/20_score_lactos.sh <bench>` on the first two images of each set
(MAX_SAMPLES=2) against the extracted asset bundle and compares every dump with
the lab's reference dumps (the ones the paper's numbers were computed from).

Needs: a GPU, the bundle at $ASSETS (default <repo>/assets), and the lab tree
(`LAB_REPO`, default the jhong392 checkout). Skipped when either is absent.
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
LAB = Path(os.environ.get("LAB_REPO", "/usr/scratch/jhong392/workspace/Statistical_OOD_detection"))

# bench -> [(our dump subdir, lab dump dir, lab file prefix)]
CASES = {
    "CT": [("CT/sp100_m5_hs0.5_b16x64/cta", LAB / "results_eval/ct_xray_sweep/maps/sp100_m5_hs0.5_b16x64/cta", "cta__"),
           ("CT/sp100_m5_hs0.5_b16x64/ctn", LAB / "results_eval/ct_xray_sweep/maps/sp100_m5_hs0.5_b16x64/ctn", "ctn__")],
    "xray": [("xray/sp10_m20_hs0_rgbonly_b16/xrsc", LAB / "results_eval/ct_xray_sweep/maps/sp10_m20_hs0_rgbonly_b16/xrsc", "xrsc__"),
             ("xray/sp10_m20_hs0_rgbonly_b16/xrgd", LAB / "results_eval/ct_xray_sweep/maps/sp10_m20_hs0_rgbonly_b16/xrgd", "xrgd__")],
    "ffhq": [("ffhq/sp20_m20_hs1_b16x32/faces", LAB / "results_eval/ffhq_sweep/maps/sp20_m20_hs1_b16x32/faces", ""),
             ("ffhq/sp20_m20_hs1_rgbonly_b16/faces", LAB / "results_eval/ffhq_sweep/maps/sp20_m20_hs1_rgbonly_b16/faces", ""),
             ("ffhq/sp20_m20_hs1_b16x32/moreface", LAB / "results_eval/ffhq_sweep/maps/sp20_m20_hs1_b16x32/moreface", ""),
             ("ffhq/sp20_m20_hs1_rgbonly_b16/faces_normal", LAB / "results_eval/ffhq_sweep/maps/sp20_m20_hs1_rgbonly_b16/faces_normal", "")],
}

pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not (ASSETS / "checkpoints").is_dir(), reason="asset bundle not extracted"),
    pytest.mark.skipif(not (LAB / "results_eval/ct_xray_sweep/maps").is_dir(), reason="lab reference dumps not reachable"),
]


def _run(bench, out):
    env = dict(os.environ, ASSETS=str(ASSETS), OUT=str(out), MAX_SAMPLES="2", GPU=os.environ.get("GPU", "0"))
    r = subprocess.run(["bash", str(ROOT / "scripts" / "20_score_lactos.sh"), bench], cwd=ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out / f"{bench}.log").write_text(r.stdout)
    assert r.returncode == 0, r.stdout[-4000:]


@pytest.mark.parametrize("bench", list(CASES))
def test_scorer_reproduces_paper_dumps(bench, tmp_path):
    _run(bench, tmp_path)
    for sub, lab_dir, prefix in CASES[bench]:
        ours = sorted(glob.glob(str(tmp_path / "lactos_maps" / sub / "samples_*.npz")))
        assert len(ours) == 2, f"{sub}: expected 2 dumps, found {ours}"
        for f in ours:
            ref_path = lab_dir / (prefix + os.path.basename(f))
            assert ref_path.exists(), f"no lab reference {ref_path}"
            a, b = np.load(f), np.load(ref_path)
            assert (a["gt_mask"] > 0).astype(np.uint8).tolist() == (b["gt_mask"] > 0).astype(np.uint8).tolist(), f"{sub}/{f}: gt_mask differs"
            # an empty valid_mask means "every pixel valid" (full_metric_table.py convention);
            # the lab's CT/X-ray dumps store an explicit all-ones mask instead
            full = lambda v: (v > 0) if v.shape == a["delta_map"].shape else np.ones(a["delta_map"].shape, bool)
            assert full(a["valid_mask"]).tolist() == full(b["valid_mask"]).tolist(), f"{sub}/{f}: valid_mask differs"
            da, db = a["delta_map"].astype(np.float64), b["delta_map"].astype(np.float64)
            assert np.array_equal(np.isnan(da), np.isnan(db)), f"{sub}/{f}: NaN pattern differs"
            m = ~np.isnan(da)
            diff = np.abs(da[m] - db[m]).max() if m.any() else 0.0
            assert diff <= 1e-4, f"{sub}/{os.path.basename(f)}: max |delta diff| = {diff:.3g}"
