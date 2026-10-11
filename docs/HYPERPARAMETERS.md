# Hyper-parameters, cell by cell

## LACToS (`scripts/20_score_lactos.sh` → `evaluate.py`)

Shared by every run: ResNet-101 (torchvision `IMAGENET1K_V2`) layer1–3 features with 3×3 patch
context, L2-normalised per layer and concatenated (1792 dims); a per-dataset PixelAutoEncoder
compresses them to 3 dims (`--n_pca 3`, `--autoencoder_path`); SLIC superpixels
(`super_pixel_generation.py`, 50 segments) are recursively refined to the target size
(`--superpixel_target_size`); per superpixel, one histogram over the RGB values and one over the
3 latent values of all reconstruction pixels (`--bins_rgb`, `--bins_pca`), optionally smoothed
with a Gaussian of `--smooth_sigma` bins; the score is (NLL − entropy) of the test image's pixels
under each histogram, intensity term + λ · feature term; the per-superpixel map is smoothed with a
spatial Gaussian (stage 4).

| | CHAOS | FFHQ | DVXray |
|---|---|---|---|
| reconstructions used | 5 draws (`recons/..._CT_{cta,ctn}`) | 20 seeds | 20 draws |
| `--superpixel_target_size` | 100 | 20 | 10 |
| `--smooth_sigma` (histogram) | 0.5 | 1 | 0 |
| `--bins_rgb` × `--bins_pca` | 16 × 64 | 16 × 32 | 16 × 16 |
| λ (stage 4) | 1 | 0.25 (full dump + `--no_pca` dump) | 0 (`--no_pca` dump only) |
| spatial σ (stage 4) | 2 | 5 | 5 |
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
the paper was computed with `folded` (the lab's package), the FFHQ row with `direct`. With
`direct` on CHAOS the pooled metrics are (filled in by `docs/REPRODUCIBILITY.md` after the
verification run).

## Baselines (`scripts/30_run_baselines.sh` → `baselines/infer_*.py`, composed by `baselines/compose.py`)

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
