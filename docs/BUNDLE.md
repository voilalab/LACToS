# The asset bundle `LACToS_assets`

Download: [`LACToS_assets.tar`](https://drive.google.com/file/d/1o5bz1IcF077sUJUqgsZIuAWurfJMjtsj/view?usp=sharing)
(Google Drive, 7.5 GB). Extract anywhere and either symlink it to `<repo>/assets` or set
`ASSETS=/path/to/LACToS_assets` for every script. `bash scripts/check_assets.sh` checks
that the extracted tree is complete (required files, image and reconstruction counts).

```
LACToS_assets/
├── checkpoints/
│   ├── lactos/
│   │   ├── ae_CT.pth, ae_faces.pth, ae_xray.pth   PixelAutoEncoder 1792 -> 3 (state dicts, 8 MB each)
│   │   └── backbone_xray_ddad_lastlayer_feat4.pth ResNet-101 with layer3 fine-tuned (DVXray)
│   ├── diffusion/
│   │   ├── ct_model050000.pt                   guided-diffusion UNet, CHAOS prior          (267 MB)
│   │   ├── xray_model500000.pt                 guided-diffusion UNet, DVXray prior         (267 MB)
│   │   └── ffhq_10m.pt                         guided-diffusion FFHQ model (DPS2022)       (357 MB)
│   └── baselines/{ffhq,ct,xray}/
│       ├── patchcore/{nnscorer_search_index.faiss, patchcore_params.pkl}
│       ├── simplenet/ckpt.pth
│       ├── supersimplenet/weights.pt
│       ├── efficientad/{teacher,student,autoencoder}_final.pth, norm_stats.pt, run_args.txt
│       ├── glass/ckpt_best.pth
│       └── ddad/unet.pth
├── recons/                                     the reconstructions Table 1 was computed from
│   ├── results_patches_ddad_native_CT_cta/samples_image_XXXX/ddad_native_0_4/inpainting/{recon/{0..4}_0_00000.png, label/0_00000.png, input/0_00000.png}
│   ├── results_patches_ddad_native_CT_ctn/samples_image_6XXX/...                       (normal slices, ids 6000+i)
│   ├── results_patches_ddad_native_xray_xrsc/samples_<patient>_OL/.../recon/{0..19}_0_00000.png
│   ├── results_patches_ddad_native_xray_xrgd/...
│   ├── results_patches_adm_dps_faces/samples_<id>/adm_dps_whole_0_4/inpainting/recon/{0..19}_0_00000.png   (22 ids)
│   ├── results_patches_adm_dps_moreface/...                                              (28 ids)
│   └── results_patches_adm_dps_faces_normal/...                                          (50 ids)
├── reference_results/                          stage-2 outputs of the maintainers' verification run (docs/REPRODUCIBILITY.md)
│   ├── lactos_maps/<bench>/<tag>/<set>/samples_<id>.npz          544 LACToS score maps (what scripts/score_lactos.sh writes)
│   ├── baseline_maps/<bench>/<method>/<setting>/<set>__<id>.npz  2,664 baseline maps (what scripts/run_baselines.sh writes)
│   └── table1.{json,md}                        Table 1 computed from them: 63/63 cells equal to the paper
└── data/
    ├── CT/test/{cta,ctn}/*.png, ground_truth/{cta,ctn}/*_mask.png
    ├── CT/masks/{masks_ood_med_med,masks_body_ood_default_large}/image_{XXXX,6XXX}_mask.png   (LACToS: lesion + body masks)
    ├── CT/body_masks_baselines/image_6XXX_mask.png                                            (baselines: body masks of the normal slices)
    ├── CT/grid_subset_selection.json
    ├── xray/test/{xrsc,xrgd}/*.png, ground_truth/{xrsc,xrgd}/*_mask.png, grid_subset_selection.json
    └── ffhq/{faces,moreface,faces_normal}/{test/*.png, ground_truth/*_mask.png}, faces_normal/SOURCE.json
```

| | files | size |
|---|---:|---:|
| checkpoints | 46 | 5.8 GB |
| recons | ~7,900 PNG | 0.5 GB |
| reference_results | 3,208 npz + 2 | 0.7 GB |
| data | ~1,900 PNG + JSON | 40 MB |
| **total** | 11,413 | **7.5 GB** |

The archive is an uncompressed tar (its contents are PNG, compressed npz and PyTorch files, which do not compress further).
Every file is a plain copy; there are no symlinks inside.
