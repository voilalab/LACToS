# Verification record

## 2026-10-10 — development run (branch worktree + bundle), all 63 cells

`scripts/20_score_lactos.sh` (3 GPUs in parallel, ~45 min), `scripts/30_run_baselines.sh` (1 GPU,
~25 min) and `scripts/40_make_table1.py` on the staged bundle:

```
63/63 cells agree at 4 decimals.   PASS
```

Regression checks run before the full runs (maintainers' tools against the lab's reference dumps,
the files the paper's numbers were computed from; not part of this branch):

| test | what it checks | result |
|---|---|---|
| LACToS dump check | LACToS delta maps of the first 2 images of every set vs the lab dumps | CT, X-ray, FFHQ: max abs diff <= 1e-4 (CT needs `--ae_encode folded`) |
| baseline map check | published-blur maps of all 6 baselines, 2 images per set, vs the lab raw maps | r = 1.000000 and max diff 0 for all methods, except PatchCore max diff 2e-5 (GPU k-NN vs CPU faiss) |
| sampler smoke tests (1 image, 1 draw) | `scripts/10_sample_*.sh` run end to end | FFHQ draw = shipped draw up to 1 gray level; CHAOS / DVXray draws differ (unseeded reverse-step noise, see `sampling/README.md`) |

Hardware / software of the run: NVIDIA RTX A6000 (48 GB), CUDA 11.8 driver stack, Python 3.10.20,
torch 2.0.1+cu118, torchvision 0.15.2, scikit-image 0.19.2, scikit-learn 1.2.2, scipy 1.10.1
(`environment/ddad.txt`); baselines in `environment/{patchcore,simplenet,ssn_env}.txt`.

## What is deterministic and what is not

* Stages 2-4 from the shipped reconstructions are deterministic up to GPU floating point: two runs on
  the same GPU model give identical dumps; a different GPU or torch build can move a few histogram
  bins (the same effect as `--ae_encode`, i.e. fourth-decimal changes). Report the table your run
  produces together with `results/table1.md`'s diff.
* `--ae_encode direct` on CHAOS gives AUROC 0.9620 / AP 0.4318 / AUPRO 0.8765 instead of 0.8764.
* Stage 1 (re-sampling): FFHQ reproduces the shipped reconstructions; CHAOS / DVXray give new draws.

## 2026-10-10 — clean-clone verification (commit b7f25fc)

`git clone -b release/lactos-table1` into an empty directory (no `assets` entry in the tree), then with
`ASSETS=/path/to/LACToS_table1_assets` exported and nothing else from the lab reachable through the
repository:

```
bash scripts/fetch_third_party.sh          # 4 pinned clones
bash scripts/00_check_assets.sh            # bundle OK (8,203 files, all checksums)
GPU=0 bash scripts/20_score_lactos.sh CT & GPU=1 ... xray & GPU=2 ... ffhq & GPU=3 bash scripts/30_run_baselines.sh & wait
python scripts/40_make_table1.py
```

Result after 42 minutes of wall time on four RTX A6000s:

```
63/63 cells agree at 4 decimals.   PASS
```

`results/table1.md` of that run lists every cell with delta 0.0000 against `expected/table1.json`.
