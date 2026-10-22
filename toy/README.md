# Toy experiments (paper Section 3, Appendices B and C)

Two synthetic benchmarks on which the inlier distribution is controlled exactly. Every image holds
four grey shapes, one per quadrant; the top-right object is always a circle in the inlier
distribution, and the anomaly is a square there.

* **Toy 1, multimodality** (Figure 2, Appendix Figure 7): the vocabulary of the other three
  quadrants grows from {circle, square} (level 1) to five classes (level 4). The figure of merit
  is the false-positive rate on the other three objects at a fixed detection rate of the anomaly.
* **Toy 2, rare inlier features** (Appendix Figure 9): the vocabulary stays {circle, square} while
  P(square) at the other quadrants falls from 0.5 to 0.1. Two AUROCs: in-distribution squares
  against in-distribution circles (is a rare but normal feature flagged?), and the anomalous square
  against identical normal squares (is the anomaly still singled out?).

Each dataset has its own diffusion prior, a DDAD UNet trained here with `toy/train_prior.py`; the
priors are not in the asset bundle. LACToS and DDAD score the same DPS reconstructions, PatchCore
(toy 1 only) fits its memory bank on the training images.

## Steps

```bash
conda activate lactos                                   # README "Installation"
python toy/gen_data.py                                  # 9 datasets, 5,300 images each (~10 min, CPU); toy/data/
GPUS="0 1 2 3 4 5 6 7" bash toy/train_priors.sh         # 8 priors, one per GPU, ~8.5 h each (RTX 6000 Ada); toy/checkpoints/<dataset>/250
GPUS="0 1 2 3" bash toy/run_toy1.sh                     # ~2 h: reconstructions, scoring, PatchCore, figures; results/toy/quad/
GPUS="0 1 2 3" bash toy/run_toy2.sh                     # ~3 h; results/toy/quadp/
```

`run_toy1.sh` needs the `patchcore` environment and `third_party/patchcore-inspection` (README,
stage 2b); point it at the interpreter with `PY_PATCHCORE=...` if the environment is not under
conda's `envs/`. Both run scripts are resumable: finished reconstruction banks and map folders are
skipped. All scripts take `OUT=` (results, default `results/toy`), `DATA=` and `CKPT=`.

Smoke tests, a few minutes each on one GPU:

```bash
python -m pytest toy/tests                                                        # renderer, generator, scoring helpers
CUDA_VISIBLE_DEVICES=0 python toy/train_prior.py --dataset quad_L1 --epochs 1 --save_every 1 --max_images 64 --out results/toy/smoke_ckpt
GPUS="0" LEVELS="1" NANOM=2 NGOOD=2 NSEEDS=2 KS="1 2" MAX_TRAIN=200 OUT=results/toy_smoke bash toy/run_toy1.sh
GPUS="0" DATASETS="quadp_p50" NANOM=2 NSEEDS=2 KS="1 2" NMEAN=2 OUT=results/toy_smoke bash toy/run_toy2.sh
```

## What each script does

| Script | Role |
|---|---|
| `gen_data.py`, `shapes.py`, `config.yaml` | anti-aliased shape renderer and the nine datasets (seeds fixed; `quadp_p50` is `quad_L1`'s law and links its images) |
| `train_prior.py`, `ddad_config.yaml` | DDAD's training loop (UNet 256/64, epsilon loss, Adam 3e-4, batch 32, 1000-step linear schedule, seed 42) |
| `sampling/generate_recons.py` (`model_backbone: ddad`) | DPS reconstructions: y = x + σn, scale 0.5, 1000 steps skip 25, η = 1; PNG and float32 |
| `prepare_superpixels.py`, `score_from_recons.py` | SLIC masks; LACToS and DDAD maps from one reconstruction bank (`--mode single / mean / set`, `--k`) |
| `run_patchcore.py` | PatchCore (WideResNet-50 layers 2+3, 10% coreset, 256 px, no crop) on train/good, maps of both test folders |
| `analyze_toy1.py` | per-image FPR on the other objects at TPR 80/90/95/99 %, bootstrap CIs, markdown table, labelled and bare panels |
| `analyze_toy2.py` | pooled-pixel AUROCs against p, bootstrap CIs, markdown table, labelled figure and bare panels |
| `run_toy1.sh`, `run_toy2.sh`, `train_priors.sh`, `common.sh` | orchestration, one job per GPU |

## Settings

| | Toy 1 | Toy 2 |
|---|---|---|
| datasets | `quad_L1` .. `quad_L4` | `quadp_p50` .. `quadp_p10` |
| test images | 50 with a square at Q1, first 50 anomaly-free | the same 50 for every p: a square at Q1 and at least one at Q2-Q4 |
| DPS σ, seeds | 0.30, 20 | 0.10, 40 |
| LACToS | set of 1 / 5 / 10 / 20 reconstructions | set of 1 / 2 / 5 / 10 / 20 / 40 |
| DDAD | 1 sample, mean of 20 | 1 sample, mean of 20 (first panel) |
| PatchCore | yes | no |
| figure | FPR at TPR ∈ {80, 90, 95, 99} % | pooled AUROC, two questions |

LACToS scores the reconstructions with ResNet-101 features projected on a 5-dim PCA basis of the
test image, superpixels of ~400 px, 32 intensity and 16 feature bins and histogram σ 0.1; the map
is blurred with σ 4 before evaluation to match the σ 4 blur inside DDAD's and PatchCore's
pipelines. DDAD uses its frozen WideResNet-101-2 features and v = 1 on the float32
reconstruction.

## Output

```
results/toy/quad/L<N>/recons/<cell>/samples_<id>/ddad_dps_whole_0_4/inpainting/{recon,recon_float,label}/   anomalous bank
results/toy/quad/L<N>/good/recons/<cell>/...                                                              anomaly-free bank
results/toy/quad/L<N>/[good/]<cell>/{single,mean20,set1,set5,set10,set20}/maps/{ddad,lactos}/<id>.npy
results/toy/quad/L<N>/patchcore/{q1_square,good}/maps/patchcore/<id>.npy
results/toy/quad/toy1_fpr_additive_dps_s030.{json,md}, toy1_fpr_additive_dps_s030_tpr95.png, ..._bare.png
results/toy/quadp/<dataset>/recons/<cell>/..., <dataset>/<cell>/{single,mean20,set1,...,set40}/maps/{ddad,lactos}/
results/toy/quadp/toy2_auroc_additive_dps_s010.{json,md,png}, ..._id_bare.png, ..._q1_bare.png
```

The paper's Figures 2, 7 and 9 are the `_bare` panels with the example images, axis labels and
legends laid out separately; the labelled PNGs carry the same curves with the paper's legend names.

<!-- ## What reproduces exactly and what does not

The datasets are deterministic: every image is regenerated byte for byte from its seed. Training
is not: GPU kernels make two trainings of the same prior differ, DPS then draws different
reconstructions, and the curves move within their bootstrap bands. The ordering and the trends
of the figures are what to expect to reproduce, not the third decimal.

Given a prior, the reconstructions are seeded per image and seed. Re-sampling the paper's banks
with the paper's priors reproduced most draws to one grey level, but a few draws of a stochastic
trajectory land in another mode (a circle where the original draw had a square): round-off is
enough to tip them. The scoring and the analysis are deterministic given the maps, and
`analyze_toy1.py` / `analyze_toy2.py` reproduce the paper's statistics exactly from the paper's maps. -->
