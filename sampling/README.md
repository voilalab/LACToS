# Stage 1: sampling the reconstructions

LACToS scores a test image against M posterior samples of it, drawn with diffusion posterior
sampling (DPS) under the identity forward model y = x + σn. The asset bundle contains the
reconstructions the paper scored (`recons/`); the scripts here are how they are produced.

```bash
GPUS="0 1 2 3" bash scripts/sample_ct.sh       # CHAOS   200 slices x  5 draws = 1,000 reconstructions
GPUS="0 1 2 3" bash scripts/sample_xray.sh     # DVXray  144 images x 20 draws = 2,880 reconstructions
GPUS="0 1 2 3" bash scripts/sample_ffhq.sh     # FFHQ    100 faces  x 20 seeds = 2,000 reconstructions
```

Every script starts one process per GPU in `GPUS` (or uses `GPU=<index>`),
writes one log per process to `results/logs/sample_*_shard<i>.log`, and resumes when re-run:
reconstructions already on disk are skipped. A set can be named to run it alone
(`scripts/sample_ct.sh cta`, `... xray.sh xrgd`, `... ffhq.sh faces_normal`). Smoke tests:

```bash
GPU=0 IMAGES=image_0000 DRAWS=1 bash scripts/sample_ct.sh cta
GPU=0 IMAGES=P01809_OL  DRAWS=1 bash scripts/sample_xray.sh xrsc
GPU=0 SAMPLES="40" SEEDS=1      bash scripts/sample_ffhq.sh faces
```

## Output and what to do with it

```
results/recons_mine/
  results_patches_ddad_native_CT_{cta,ctn}/samples_image_XXXX/ddad_native_0_4/inpainting/{recon/<k>_0_00000.png, label/0_00000.png, input/0_00000.png}
  results_patches_ddad_native_xray_{xrsc,xrgd}/samples_<id>/ddad_native_0_4/inpainting/...
  results_patches_adm_dps_{faces,moreface,faces_normal}/samples_<id>/adm_dps_whole_0_4/inpainting/...
  raw/                                             sample_fast.py output before re-layout (CHAOS, DVXray)
```

The seven `results_patches_*` directories have the names and layout of `recons/` in the bundle,
which is never modified by these scripts.

Score them with `RECON_ROOT=results/recons_mine GPU=0 bash scripts/score_lactos.sh`, then
`python scripts/make_table1.py --only lactos`.

## Samplers and settings (paper Table 5)

| | CHAOS | DVXray | FFHQ |
|---|---|---|---|
| script | `sample_ct.sh` | `sample_xray.sh` | `sample_ffhq.sh` |
| sampler | `sampling/dps/sample_fast.py` | `sampling/dps/sample_fast.py` | `sampling/generate_recons.py` (`additive_dps`) |
| noise σ | 0.1 | 2.0 | 0.01 |
| DPS scale | 0.4 (ζ′ = 0.2) | 1.0 (ζ′ = 0.5) | 0.1 |
| steps | 1000 DDPM | 1000 DDPM | 1000 (skip 1), η = 1 |
| draws M | 5 | 20 | 20 (seeds 0–19) |
| prior (`checkpoints/diffusion/`) | `ct_model050000.pt` | `xray_model500000.pt` | `ffhq_10m.pt` |

* `sampling/dps/` is the DPS implementation used for CHAOS and DVXray: the `guided_diffusion`
  package of the DPS release plus `sample_fast.py`, a batched sampler for the identity forward
  model with the per-sample residual normalisation of the DPS paper (ζ_i = ζ′ / ‖y − A(x̂₀)‖;
  `--scale` = 2ζ′). Configs: `configs/ctgrid/sigma0.1.yaml`, `configs/xray_inpainting_sigma2.yaml`,
  `ct_model_config.yaml` / `xray_model_config.yaml` (UNet architecture; the checkpoint is passed
  with `--model_path`). `sampling/wire_recons.py` moves its output into the layout above.
* `sampling/generate_recons.py` with `ood/samplers/additive_dps.py` runs DPS on the guided-diffusion
  FFHQ model (`DDAD_DPS/samplers.py` implements the reverse process). Configs:
  `configs/recon/adm_dps_ffhq_*.yaml`.

## Determinism

* **FFHQ** is seeded per draw (`torch.manual_seed(seed)` before each reverse process). A re-run
  reproduces the bundled reconstructions up to PNG rounding: at most one gray level, on about 1 %
  of the pixels (checked on four draws).
* **CHAOS and DVXray**: `sample_fast.py` seeds the measurement noise and the initial x_T per
  (image, draw), but the noise of its 1000 reverse steps comes from the global torch RNG. Every run
  therefore gives different posterior draws of the same model (one draw differs from the bundled
  one by 4–5 gray levels on average). The sampler is kept exactly as it produced the paper's
  reconstructions.

How much new draws move the LACToS cells of Table 1, measured with alternative draws that the same
sampler produced (pooled AUROC / AP / AUPRO, scored on one machine):

| | draws | AUROC | AP | AUPRO |
|---|---|---:|---:|---:|
| CHAOS (M = 5) | the paper's 5 draws | 0.9616 | 0.4295 | 0.8751 |
| | 5 other draws | 0.9621 | 0.4090 | 0.8658 |
| DVXray (M = 10) | first 10 of the paper's 20 draws | 0.9585 | 0.0530 | 0.8055 |
| | the other 10 draws | 0.9581 | 0.0519 | 0.8079 |

(These rows were scored on CPU, which itself shifts the CHAOS cell from 0.9620 / 0.4318 / 0.8764
by a few units in the fourth and third decimal; see `docs/REPRODUCIBILITY.md`.) With re-sampled
CHAOS or DVXray reconstructions, `scripts/make_table1.py` reports the LACToS cells as `DIFF`
with respect to the paper; the bundled reconstructions are the ones behind the published numbers.
