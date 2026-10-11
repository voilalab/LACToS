# Stage 1: sampling the reconstructions (optional)

The bundle ships every reconstruction Table 1 was computed from. These scripts regenerate them.

```bash
GPU=0 bash scripts/10_sample_ct.sh            # CHAOS: 100 + 100 slices x 5 draws  (~2 GPU-h on an A6000)
GPU=0 bash scripts/10_sample_xray.sh          # DVXray: 72 + 72 bags x 20 draws     (~4 GPU-h)
GPU=0 bash scripts/10_sample_ffhq.sh          # FFHQ: 100 faces x 20 seeds          (~8 GPU-h)
```

Parallelise CHAOS / DVXray over GPUs with `SHARD=i NSHARDS=n` (draws are seeded per image and
draw index, so shards and resumed runs give the same files). Smoke tests:

```bash
GPU=0 IMAGES=image_0000 DRAWS=1 bash scripts/10_sample_ct.sh cta
GPU=0 IMAGES=P01809_OL  DRAWS=1 bash scripts/10_sample_xray.sh xrsc
GPU=0 SAMPLES="40" SEEDS=1      bash scripts/10_sample_ffhq.sh faces
```

Output goes to `results/recons_mine/` in the layout `evaluate.py` reads; the shipped `assets/recons/`
is never modified. Score your own reconstructions with

```bash
RECON_ROOT=results/recons_mine GPU=0 bash scripts/20_score_lactos.sh CT      # or xray | ffhq
python scripts/40_make_table1.py --only lactos
```

## Samplers

* `sampling/dps/` — the DPS implementation used for CHAOS and DVXray: the DPS2022 `guided_diffusion`
  package plus `sample_fast.py`, a batched sampler for the identity forward model (y = x + σn) with
  the paper's per-sample residual normalisation (ζ_i = ζ′ / ‖y − A(x̂₀)‖; `--scale` = 2ζ′). Configs:
  `configs/ctgrid/sigma0.1.yaml` (CHAOS), `configs/xray_inpainting_sigma2.yaml` (DVXray),
  `ct_model_config.yaml` / `xray_model_config.yaml` (UNet architecture; the checkpoint path is
  passed with `--model_path`).
* `sampling/generate_recons.py` with `ood/samplers/additive_dps.py` — DPS on the guided-diffusion
  FFHQ model (`DDAD_DPS/samplers.py` implements the reverse process). Configs: `configs/recon/adm_dps_ffhq_*.yaml`.
* `sampling/wire_recons.py` — converts `sample_fast.py` output (`<stem>/recon/m<k>.png`) into the
  `samples_<id>/ddad_native_0_4/inpainting/{recon,label,input}` layout.

## Determinism

The FFHQ sampler (`generate_recons.py`) seeds the whole reverse process per draw
(`torch.manual_seed`); a re-run reproduces the shipped reconstructions up to PNG rounding (smoke
test: max 1 gray level, 0.8 % of pixels). `sample_fast.py` (CHAOS, DVXray) seeds only the
measurement noise and the initial x_T per (image, draw); the noise of its 1000 reverse steps comes
from the global torch RNG, so a re-run gives **different posterior draws** of the same model
(smoke test: mean |diff| 3.6 - 4.8 gray levels). The sampler is shipped exactly as it produced the
paper's reconstructions. Re-sampled CHAOS / DVXray sets therefore give Table 1 numbers that differ
in the last decimals from the shipped ones; the reconstructions in the bundle are the ones Table 1 was computed from and the ones to
use for an exact reproduction.
