# Verification record

## 2026-10-10 — development run (branch worktree + bundle), all 63 cells

`scripts/score_lactos.sh` (3 GPUs in parallel, ~45 min), `scripts/run_baselines.sh` (1 GPU,
~25 min) and `scripts/make_table1.py` on the staged bundle:

```
63/63 cells agree at 4 decimals.   PASS
```

Regression checks run before the full runs (maintainers' tools against the lab's reference dumps,
the files the paper's numbers were computed from; not part of this repository):

| test | what it checks | result |
|---|---|---|
| LACToS dump check | LACToS delta maps of the first 2 images of every set vs the lab dumps | CT, X-ray, FFHQ: max abs diff <= 1e-4 (CT needs `--ae_encode folded`) |
| baseline map check | published-blur maps of all 6 baselines, 2 images per set, vs the lab raw maps | r = 1.000000 and max diff 0 for all methods, except PatchCore max diff 2e-5 (GPU k-NN vs CPU faiss) |
| sampler smoke tests (2 images, 2 draws, 2 GPUs) | `scripts/sample_*.sh`, then `RECON_ROOT=... scripts/score_lactos.sh` | all run end to end; FFHQ draws equal the bundled ones up to 1 gray level; CHAOS / DVXray draws differ (unseeded reverse-step noise, see `sampling/README.md`) |

Hardware / software of the run: NVIDIA RTX A6000 (48 GB), CUDA 11.8 driver stack, Python 3.10.20,
torch 2.0.1+cu118, torchvision 0.15.2, scikit-image 0.19.2, scikit-learn 1.2.2, scipy 1.10.1
(`environment/ddad.txt`); baselines in `environment/{patchcore,simplenet,ssn_env}.txt`.

## What is deterministic and what is not

* Stages 2-4 from the bundled reconstructions are deterministic on a given machine: two runs on the
  same GPU model give identical dumps. Other hardware changes float32 round-off, which moves a few
  values across histogram-bin edges. Measured for the CHAOS LACToS cell (0.9620 / 0.4318 / 0.8764 on
  an RTX A6000): 0.9616 / 0.4295 / 0.8751 when scored on CPU (`evaluate.py --device cpu`), and
  0.9620 / 0.4318 / 0.8765 with `--ae_encode direct`. Report the table your run produces together
  with the diff in `results/table1.md`.
* Stage 1 (sampling): FFHQ reproduces the bundled reconstructions up to PNG rounding; CHAOS and
  DVXray give new posterior draws on every run, which changes their LACToS cells in the second to
  third decimal (table in `sampling/README.md`).

## 2026-10-10 — clean-clone verification (commit fa7ab56, the trimmed branch)

A fresh `git clone` of this code into an empty directory (no `assets` entry in the tree), then with
`ASSETS=/path/to/LACToS_assets` exported and nothing else from the lab reachable through the
repository:

```
bash scripts/fetch_third_party.sh          # 4 pinned clones
bash scripts/check_assets.sh               # bundle OK
GPU=0 bash scripts/score_lactos.sh CT & GPU=1 ... xray & GPU=2 ... ffhq & GPU=3 bash scripts/run_baselines.sh & wait
python scripts/make_table1.py
```

Result after 41 minutes of wall time on four RTX A6000s (an earlier identical run on commit b7f25fc, before the final trim, also passed):

```
63/63 cells agree at 4 decimals.   PASS
```

`results/table1.md` of that run lists every cell with delta 0.0000 against `expected/table1.json`.
