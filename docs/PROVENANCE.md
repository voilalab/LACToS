# What the artefacts are and where they come from

## Test data (`assets/data/`)

| Set | n | What |
|---|---:|---|
| `CT/test/cta`, `ground_truth/cta` | 100 | CHAOS healthy CT slices with a synthetic tumour warped in; 256×256 grayscale; lesion masks 512×512 (subsampled 2× at evaluation) |
| `CT/test/ctn`, `ground_truth/ctn` | 100 | the same base slices without the tumour, ids `image_6000+i` so that they never collide with the anomalous slice; all-zero masks |
| `CT/masks/` | 200 + 200 | lesion masks (`masks_ood_med_med`) and body masks (`masks_body_ood_default_large`) in the layout `evaluate.py` reads for CT; ids 0000–0099 and 6000–6099 |
| `CT/body_masks_baselines/` | 100 | body masks of the normal slices used by the baseline evaluation (see `HYPERPARAMETERS.md`) |
| `xray/test/{xrsc,xrgd}`, `ground_truth/` | 72 + 72 | DvXray scissors positives / weapon-free negatives at 256×256; scissor masks from SAM 3 on the dataset's boxes, manually filtered; the negatives are held out from the 5,000 training images |
| `ffhq/faces` | 22 | curated photographs of characters with prosthetics or atypical facial features, FFHQ-style crops; ten masks are 1254×1254 (resized nearest at evaluation) |
| `ffhq/moreface` | 28 | second curated anomalous set, 256×256; shares ids 1–10 with `faces`, so the sets stay in separate directories |
| `ffhq/faces_normal` | 50 | FFHQ images 00000–00049 (Lanczos to 256), ids 1000–1049 (`SOURCE.json`); anomaly-free, all-zero masks |
| `*/grid_subset_selection.json` | | the 10 + 10 ids on which sampling-time hyper-parameters were selected |

## Reconstructions (`assets/recons/`)

| Dir | Sampler (stage 1 scripts) | Prior | Draws |
|---|---|---|---:|
| `results_patches_ddad_native_CT_{cta,ctn}` | `sampling/dps/sample_fast.py`: y = x + 0.1·n, DPS scale 0.4 (ζ′ = 0.2), 1000 steps | `ct_model050000.pt` | 5 |
| `results_patches_ddad_native_xray_{xrsc,xrgd}` | same, σ = 2.0, scale 1.0 (ζ′ = 0.5) | `xray_model500000.pt` | 20 |
| `results_patches_adm_dps_{faces,moreface,faces_normal}` | `sampling/generate_recons.py` (additive_dps): σ = 0.01, scale 0.1, skip 1, 1000 steps | `ffhq_10m.pt` | 20 |

## Checkpoints (`assets/checkpoints/`)

| File | What | Trained on |
|---|---|---|
| `lactos/ae_CT.pth`, `ae_faces.pth`, `ae_xray.pth` | PixelAutoEncoder 1792→3 on ResNet-101 layer1–3 features (bias-free 1×1 convs) | CHAOS inlier slices / FFHQ / DvXray negatives |
| `lactos/backbone_xray_ddad_lastlayer_feat4.pth` | ResNet-101 with layer3 fine-tuned 4 epochs with the DDAD domain-adaptation loss | DvXray negatives |
| `diffusion/ct_model050000.pt` | guided-diffusion UNet (96 channels), 50k steps | 3,070 CHAOS inlier slices |
| `diffusion/xray_model500000.pt` | guided-diffusion UNet (96 channels), 500k steps | 5,000 DvXray negatives |
| `diffusion/ffhq_10m.pt` | guided-diffusion FFHQ model released with DPS (Chung et al., 2023) | FFHQ |
| `baselines/<bench>/patchcore/` | faiss memory bank + params (WideResNet-50 layer2+3, 224 crop) | FFHQ 5k / CHAOS / DvXray 5k inliers |
| `baselines/<bench>/simplenet/ckpt.pth` | SimpleNet (WRN-50; 288 crop for FFHQ and DvXray, 256 for CT) | idem |
| `baselines/<bench>/supersimplenet/weights.pt` | SuperSimpleNet, 256 input | idem |
| `baselines/<bench>/efficientad/` | EfficientAD medium, final checkpoint (no test-set selection), with `norm_stats.pt` | idem |
| `baselines/<bench>/glass/ckpt_best.pth` | GLASS; **`ckpt_best` is selected on the test set by the upstream training code** | idem |
| `baselines/<bench>/ddad/unet.pth` | DDAD UNet: FFHQ epoch 2000, CHAOS epoch 2000, DvXray epoch 500 | idem |

Training code is not part of this repository.

## How the table's settings were chosen

LACToS hyper-parameters (superpixel size, draws M, histogram σ, bins, λ, spatial σ) and the
baselines' inference-time parameters (PatchCore k, EfficientAD α, DDAD (w, v) at T = 250) were
selected on the 20-image tuning subsets (`grid_subset_selection.json`; FFHQ: first 5 faces +
5 moreface + 10 normals), as described in the paper; `HYPERPARAMETERS.md` lists the selected
values and every flag.
