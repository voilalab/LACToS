# Provenance of every artefact behind Table 1

Written 2026-10-10 while assembling this branch. Lab paths are given so that a maintainer can
trace a file back; a user of the bundle needs none of them.

## Test data (`assets/data/`)

| Set | n | Source | Notes |
|---|---:|---|---|
| `CT/test/cta`, `ground_truth/cta` | 100 | CHAOS healthy slices with a synthetic tumour warped in (lab: `handoff_CT_xray_20260924/CT/test/anomalous`) | 256×256 grayscale; lesion masks 512×512, subsampled 2× at evaluation |
| `CT/test/ctn`, `ground_truth/ctn` | 100 | the same base slices without the tumour (`.../CT/test/normal`), **renamed `image_6000+i`** so that ids never collide with the anomalous slice | all-zero masks |
| `CT/masks/masks_body_ood_default_large` | 200 | body masks used by the **LACToS** evaluation (`OOD_reproduction_package/data/id_new_warped_images`) | ids 0000–0099 identical to the handoff masks; ids 6000–6099 are a different (image-derived) segmentation, IoU 0.944–0.985 with the handoff masks |
| `CT/masks/masks_ood_med_med` | 200 | lesion masks in the layout `evaluate.py` reads for CT (`OOD_CT_MASK_ROOT`) | same content as `ground_truth/` |
| `CT/body_masks_baselines` | 100 | body masks used by the **baseline** evaluation of the normal slices (`handoff .../CT/body_masks`) | see the caveat in the README |
| `xray/test/{xrsc,xrgd}`, `ground_truth/` | 72 + 72 | DvXray scissors positives / weapon-free negatives at 256×256 (`handoff .../xray`); scissor masks from SAM 3 on the dataset's boxes, manually filtered | the 72 negatives are **training images** of every model |
| `ffhq/faces` | 22 | manually curated photographs of characters with prosthetics / atypical features, FFHQ-style crops (lab `images/faces`, ids 5 and 98 dropped 2026-09-22) | ten masks are 1254×1254 and are resized (nearest) at evaluation |
| `ffhq/moreface` | 28 | second curated anomalous set (lab `images/moreface_masked_subset_256_fullsized_masks_binary`) | shares ids 1–10 with `faces`; sets are always kept in separate directories |
| `ffhq/faces_normal` | 50 | FFHQ images 00000–00049 (Lanczos to 256), ids 1000–1049 (`SOURCE.json`) | anomaly-free; all-zero masks |
| `*/grid_subset_selection.json` | | the 10 + 10 ids on which sampling-time hyper-parameters were selected | FFHQ had no recorded subset (first 5 faces + 5 moreface + 10 normals were used) |

## Reconstructions (`assets/recons/`)

| Dir | Sampler | Prior | Draws | Lab source |
|---|---|---|---:|---|
| `results_patches_ddad_native_CT_cta` / `_ctn` | `sampling/dps/sample_fast.py`, y = x + 0.1·n, scale 0.4 (ζ′ = 0.2), 1000 steps, seed 3 | `ct_model050000.pt` | 5 (first 5 of 10) | `dps_ct_bundle/outputs/CT_sigma0.1_zeta0.2_full`, `ID_same_base_sigma0.1_zeta0.2` |
| `results_patches_ddad_native_xray_xrsc` / `_xrgd` | same sampler, σ = 2.0, scale 1.0 (ζ′ = 0.5) | `xray_model500000.pt` | 20 | `dps_ct_bundle/outputs/XRAY_{scissors,good}_zeta0.5_sigma2.0_M20` |
| `results_patches_adm_dps_{faces,moreface,faces_normal}` | `sampling/generate_recons.py` (additive_dps), σ = 0.01, scale 0.1, skip 1, 1000 steps | `ffhq_10m.pt` (DPS2022 release) | 20 | lab `results_patches_adm_dps_*` |

`sample_fast.py` seeds each (image, draw) pair independently (seed·1000003 + index·1009 + k), so a
5-draw run reproduces the first 5 draws of the 10-draw run.

## Checkpoints (`assets/checkpoints/`)

| File | What | Trained on / origin |
|---|---|---|
| `lactos/ae_CT.pth` | PixelAutoEncoder 1792→3 on ImageNet ResNet-101 layer1–3 features | CHAOS inlier slices (`OOD_reproduction_package/models/pixel_autoencoder_CT.pth`) |
| `lactos/ae_faces.pth` | idem | FFHQ (lab `checkpoints/faces/autoencoder/faces.pth`; **not** the `pixel_autoencoder_faces.pth` of the older per-image table) |
| `lactos/ae_xray.pth` | idem, on the fine-tuned backbone below | DvXray negatives (`all_categories_embedder/models/xray_mine/last_layer.pth`); unused by the Table 1 cell (λ = 0) but needed for λ > 0 |
| `lactos/backbone_xray_ddad_lastlayer_feat4.pth` | ResNet-101, layer3 fine-tuned 4 epochs with the DDAD domain-adaptation loss | `experiments/xray_finetune_mine/last_layer/checkpoints/feat4.pth` |
| `diffusion/ct_model050000.pt` | guided-diffusion UNet (96 ch), 50k steps | 3,070 CHAOS inlier slices |
| `diffusion/xray_model500000.pt` | guided-diffusion UNet (96 ch), 500k steps | 5,000 DvXray negatives |
| `diffusion/ffhq_10m.pt` | guided-diffusion FFHQ model | DPS2022 release (Chung et al.) |
| `baselines/<bench>/patchcore/` | faiss memory bank + params (WideResNet-50 layer2+3, 224 crop, k = 1) | FFHQ 5k / CHAOS / DvXray 5k inliers |
| `baselines/<bench>/simplenet/ckpt.pth` | SimpleNet (WRN-50, 288 crop for FFHQ and DvXray, 256 for CT) | FFHQ 70k / CHAOS / DvXray 5k |
| `baselines/<bench>/supersimplenet/weights.pt` | SuperSimpleNet, 256 input | idem |
| `baselines/<bench>/efficientad/` | EfficientAD medium, final checkpoint (no test-set selection) + `norm_stats.pt` | idem |
| `baselines/<bench>/glass/ckpt_best.pth` | GLASS, **`ckpt_best` = selected on the test set by the upstream code** | idem |
| `baselines/<bench>/ddad/unet.pth` | DDAD UNet: FFHQ epoch 2000, CHAOS epoch 2000, DvXray epoch 500 | idem |

Checkpoints for CHAOS and DvXray were trained by the authors; the FFHQ baselines by a
collaborator; none of the training code is part of this branch.

## Where the paper's numbers come from

* **LACToS**: hyper-parameters were chosen on the 20-image subsets over a grid of superpixel
  size × draws × histogram σ × bins × λ × spatial σ (lab `results_eval/{ct_xray_sweep,ffhq_sweep,subset_tuning}`,
  rule: intensity bins 16, spatial σ ≤ 5, M ∈ {5, 20}). The CHAOS and DVXray delta maps were
  computed with the lab's package copy of the scorer, FFHQ with this repository's; the two differ
  only in the AE evaluation order (`--ae_encode`), reproduced here.
* **Baselines**: lab `results_eval/baseline_sweep/main_table_subset_T250_nosigma.tex`. PatchCore k,
  EfficientAD α and DDAD (w, v) were selected on the same 20-image subsets (DDAD's T fixed at 250);
  maps at each method's published blur composition.
