# Upstream baseline implementations

`bash scripts/fetch_third_party.sh` clones these here, at the commits used for Table 1:

| Directory | Repository | Commit | Used by |
|---|---|---|---|
| `patchcore-inspection/` | https://github.com/amazon-science/patchcore-inspection | `fcaa92f` | `baselines/infer_feature.py --method patchcore` (`patchcore.patchcore`, `patchcore.common`, `patchcore.backbones`) |
| `SimpleNet/` | https://github.com/DonaldRR/SimpleNet | `351a2b8` | `--method simplenet` (`simplenet`, `backbones`) |
| `GLASS/` | https://github.com/cqylunlun/GLASS | `f788d56` | `--method glass` (`glass`, `backbones`) |
| `SuperSimpleNet/` | https://github.com/blaz-r/SuperSimpleNet | `98ab4d5` | `--method supersimplenet` (`model.supersimplenet`) |

Only inference code is imported; no file of these repositories is modified. EfficientAD needs no
upstream code (its checkpoints are pickled `nn.Module`s), DDAD's modules are vendored in
`baselines/ddad_lib/` (Mousakhan et al., https://github.com/arimousa/DDAD). Each repository keeps
its own licence.
