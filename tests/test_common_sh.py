"""scripts/common.sh must resolve the asset bundle robustly on a fresh clone (no GPU, no bundle needed).

A fake repository (copy of scripts/common.sh) and a stub bundle (just a checkpoints/ dir) are
created in a temp dir; common.sh is sourced via bash and its effect inspected.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _fake_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "common.sh", repo / "scripts" / "common.sh")
    bundle = tmp_path / "bundle"
    (bundle / "checkpoints").mkdir(parents=True)
    return repo, bundle


def _source(repo, env, cwd=None):
    cmd = 'source scripts/common.sh && echo "ASSETS=$ASSETS" && echo "LINK=$(readlink -f "$ROOT/assets")"'
    return subprocess.run(["bash", "-c", cmd], cwd=cwd or repo, env={**os.environ, **env},
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_dangling_assets_link_is_replaced_when_ASSETS_is_given(tmp_path):
    repo, bundle = _fake_repo(tmp_path)
    os.symlink("/nonexistent/lab/path/LACToS_table1_assets", repo / "assets")   # what a stale checkout could hold
    r = _source(repo, {"ASSETS": str(bundle)})
    assert r.returncode == 0, r.stdout
    assert (repo / "assets").resolve() == bundle.resolve()


def test_existing_assets_pointing_elsewhere_fails_loudly(tmp_path):
    repo, bundle = _fake_repo(tmp_path)
    other = tmp_path / "other"; (other / "checkpoints").mkdir(parents=True)
    os.symlink(other, repo / "assets")
    r = _source(repo, {"ASSETS": str(bundle)})
    assert r.returncode != 0
    assert "assets" in r.stdout and str(bundle) in r.stdout and str(other) in r.stdout


def test_relative_ASSETS_is_made_absolute(tmp_path):
    repo, bundle = _fake_repo(tmp_path)
    r = _source(repo, {"ASSETS": "../bundle"}, cwd=repo)
    assert r.returncode == 0, r.stdout
    assert f"ASSETS={bundle.resolve()}" in r.stdout
    assert (repo / "assets").resolve() == bundle.resolve()


def test_missing_bundle_gives_instructions(tmp_path):
    repo, _ = _fake_repo(tmp_path)
    r = _source(repo, {"ASSETS": str(tmp_path / "nowhere")})
    assert r.returncode == 2
    assert "LACToS_table1_assets" in r.stdout
