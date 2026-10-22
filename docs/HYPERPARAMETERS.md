# Hyper-parameters, cell by cell

## LACToS (`scripts/score_lactos.sh` → `evaluate.py`)

The values below are the defaults of `configs/lactos_{ct,xray,ffhq_*}.yaml`; the flags named in the
table override them on the command line.

Shared by every run: ResNet-101 (torchvision `IMAGENET1K_V2`) layer1–3 features with 3×3 patch
context, L2-normalised per layer and concatenated (1792 dims); a per-dataset PixelAutoEncoder
compresses them to 3 dims (`--n_pca 3`, `--autoencoder_path`); the superpixels are a
marker-controlled watershed of the image's edge map, with a regular grid of markers spaced for the
target size (`--superpixel_target_size`, `ood/superpixels.py`; the coarse SLIC mask that
`evaluate.py` also caches via `super_pixel_generation.py` does not enter the scores); per
superpixel, one histogram over the RGB values and one over the
3 latent values of all reconstruction pixels (`--bins_rgb`, `--bins_pca`), optionally smoothed
with a Gaussian of `--smooth_sigma` bins; the score is (NLL − entropy) of the test image's pixels
under each histogram, intensity term + λ · feature term; the per-superpixel map is smoothed with a
spatial Gaussian (stage 3).

| | CHAOS | FFHQ | DVXray |
|---|---|---|---|
| reconstructions used | 5 draws (`recons/..._CT_{cta,ctn}`) | 20 seeds | 20 draws |
| `--superpixel_target_size` | 100 | 20 | 10 |
| `--smooth_sigma` (histogram) | 0.5 | 1 | 0 |
| `--bins_rgb` × `--bins_pca` | 16 × 64 | 16 × 32 | 16 × 16 |
| λ (stage 3) | 1 | 0.25 (full dump + `--no_pca` dump) | 0 (`--no_pca` dump only) |
| spatial σ (stage 3) | 2 | 5 | 5 |
| backbone | torchvision ResNet-101 | torchvision ResNet-101 | `backbone_xray_ddad_lastlayer_feat4.pth` via `embedder/evaluate_ddad_fe.py` (`DDAD_FE_WEIGHTS`) |
| autoencoder | `ae_CT.pth` | `ae_faces.pth` | `ae_xray.pth` (not used at λ = 0) |
| `--ae_encode` | **folded** | direct | — |
| grayscale histogram | yes (CT branch of `evaluate.py`) | no | no |
| valid region | body mask (`OOD_CT_MASK_ROOT`) | all pixels | all pixels |
| config | `configs/lactos_ct.yaml` | `configs/lactos_ffhq_{faces,moreface,faces_normal}.yaml` | `configs/lactos_xray.yaml` |

Histogram-smoothing caveat (unchanged from the lab code): a histogram marginal is smoothed only
when `n_superpixels × n_bins ≤ 2·10⁸`; above that a sparse, unsmoothed path is taken. At the
settings above this affects no Table 1 cell (CHAOS: 64³ feature bins × ~675 superpixels is below
the limit; DVXray uses σ = 0).

### `--ae_encode folded` vs `direct`

`folded` applies the autoencoder's first 1×1 convolution to each ResNet layer at native
resolution before the per-layer L2 normalisation and upsampling; `direct` evaluates
`ae.encoder(embedder(x))` literally. They are the same function; float32 round-off differs
by ~1e-6 in the latents, which moves a few values across histogram-bin edges. The CHAOS row of
the paper was computed with `folded` (the lab's package), the FFHQ row with `direct`. Both encoders on both datasets (pooled AUROC / AP / AUPRO):

| | `folded` | `direct` |
|---|---|---|
| CHAOS | **0.9620 / 0.4318 / 0.8764** (paper) | 0.9620 / 0.4318 / 0.8765 |
| FFHQ | 0.8516 / 0.1177 / 0.5321 | **0.8515 / 0.1177 / 0.5320** (paper) |

DVXray uses λ = 0 (no autoencoder) and is unaffected.

## Baselines (`scripts/run_baselines.sh` → `baselines/infer_*.py`, composed by `baselines/compose.py`)

| Method | Input (resize → crop) | Setting dir | Post-hoc knob (Table 6) | Published blur composition |
|---|---|---|---|---|
| PatchCore | FFHQ 256→224; CT, DVXray 224 | `default` | k = 1 (mean of the k nearest memory-bank distances; 15 are saved) | σ = 4 at 224 on the upsampled patch map, resize to 256, σ = 5 |
| SimpleNet | FFHQ 329→288; CT 256; DVXray 288 | `default` | — | σ = 4 at input resolution, 256, σ = 5 |
| GLASS | FFHQ 288; CT 256; DVXray 288 | `default` | — | idem (scored with GLASS's own `_predict`, never through SimpleNet's class) |
| SuperSimpleNet | 256 | `default` | — | σ = 4, σ = 5 |
| EfficientAD | 256 | FFHQ `default` (4-px pad before upsampling, as upstream); CT, DVXray `nopad` (as the lab's CT / X-ray scripts) | α = 0.7 / 0 / 0.2 (weight of the student–teacher map; 1 − α on the autoencoder map) | σ = 5 only |
| DDAD | 256 | `w2_T250_s25_e1` (CHAOS, FFHQ), `w4_T250_s25_e1` (DVXray): conditioning w, 250 noising steps, skip 25, η = 1, frozen ImageNet feature extractor (WRN-101-2 for FFHQ, ResNet-101 otherwise), noise seeded per image | v = 1 / 4 / 4 (map = f_d + v·max f_d / max i_d · i_d) | σ = 4, σ = 5 |

Valid region: CT body masks (`data/CT/masks/masks_body_ood_default_large` for the anomalous
slices, `data/CT/body_masks_baselines` for the normal slices); all pixels elsewhere.

## Sampling (stage 1, optional)

| | CHAOS | DVXray | FFHQ |
|---|---|---|---|
| sampler | `sampling/dps/sample_fast.py` (DPS, per-sample residual norm) | idem | `sampling/generate_recons.py --sampler additive_dps` |
| forward model | y = x + σn, σ = 0.1 | σ = 2.0 | σ = 0.01 |
| DPS scale | 0.4 (ζ′ = 0.2) | 1.0 (ζ′ = 0.5) | 0.1 |
| steps | 1000 DDPM | 1000 | 1000 (skip 1), η = 1 |
| draws | 5 (paper), seed 3 | 20, seed 3 | 20 seeds (0–19) |
| prior | `ct_model050000.pt` | `xray_model500000.pt` | `ffhq_10m.pt` |

## CHAOS body masks of the normal slices

LACToS was evaluated inside `data/CT/masks/masks_body_ood_default_large/image_6XXX_mask.png`
(a body segmentation computed from the image when the normal slices were wired in), the
baselines inside `data/CT/body_masks_baselines/image_6XXX_mask.png` (the dataset's body masks);
IoU between the two is 0.944–0.985, and the anomalous slices use identical masks everywhere.
Pooled CHAOS metrics with the masks swapped (AUROC / AP / AUPRO):

| Method | as in the paper | with the other mask set |
|---|---|---|
| LACToS | 0.9620 / 0.4318 / 0.8764 | 0.9625 / 0.4444 / 0.8781 |
| PatchCore | 0.9297 / 0.0590 / 0.6811 | 0.9291 / 0.0571 / 0.6793 |
| SimpleNet | 0.7464 / 0.0193 / 0.4191 | 0.7436 / 0.0178 / 0.4120 |
| SuperSimpleNet | 0.3786 / 0.0051 / 0.0332 | 0.3789 / 0.0051 / 0.0333 |
| EfficientAD | 0.6664 / 0.0105 / 0.2513 | 0.6638 / 0.0103 / 0.2485 |
| GLASS | 0.8318 / 0.0410 / 0.5530 | 0.8325 / 0.0406 / 0.5542 |
| DDAD | 0.9707 / 0.3159 / 0.8676 | 0.9706 / 0.3035 / 0.8675 |
