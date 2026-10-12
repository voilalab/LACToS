"""Eval driver that uses a DDAD domain-adaptation FINE-TUNED ResNet-101 as the
feature extractor in the OOD typical-set pipeline (evaluate.py).

Difference vs evaluate_resnet_ae.py:
  - evaluate_resnet_ae.py uses a ResNet-AE *encoder* that already outputs a
    3-channel latent (the "final variables = 3" path, no PCA).
  - THIS driver uses the full DDAD fine-tuned ResNet-101 (multi-scale layer1/2/3
    features). Those are high-dimensional, so the pipeline's PCA reduces them to
    `--n_pca 3`. No separate autoencoder is trained — PCA IS the 3-variable
    reduction. (DDAD `feat*.pth` are standard ResNet-101 state dicts; the
    layer4/fc keys are ignored when loading into the layer1-3 feature extractor.)

How it works: monkeypatch evaluate.ResNetPixelEmbedder so that, after building
the normal embedder (ImageNet resnet101), it loads the DDAD fine-tuned weights
given by env var DDAD_FE_WEIGHTS into the feature extractor. Then run the
standard PCA path (do NOT pass --autoencoder_path).

Usage (from the repository root; see scripts/score_lactos.sh for the DVXray run):
  DDAD_FE_WEIGHTS=assets/checkpoints/lactos/backbone_xray_ddad_lastlayer_feat4.pth \
  python embedder/evaluate_ddad_fe.py --config configs/lactos_xray.yaml --category xray --subcategory xrsc ...
"""
import os
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate  # root pipeline
from ood.embeddings import ResNetPixelEmbedder as _RealEmbedder

WEIGHTS = os.environ.get("DDAD_FE_WEIGHTS", "")


def _unwrap(sd):
    """Unwrap container / DataParallel state dicts."""
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if isinstance(sd, dict) and len(sd) and next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


class DDADFinetunedEmbedder(_RealEmbedder):
    """ResNetPixelEmbedder whose backbone is overwritten with DDAD fine-tuned
    weights (layer1-3). Falls back to ImageNet weights if DDAD_FE_WEIGHTS unset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if WEIGHTS:
            sd = _unwrap(torch.load(WEIGHTS, map_location="cpu"))
            res = self.extractor.load_state_dict(sd, strict=False)
            # missing_keys should be empty (all layer1-3 params present);
            # unexpected_keys are the dropped layer4/fc params -> expected.
            n_missing = len(res.missing_keys)
            print(f"  [DDAD-FE] loaded fine-tuned backbone: {WEIGHTS}")
            print(f"  [DDAD-FE] missing={n_missing} (must be 0)  "
                  f"unexpected={len(res.unexpected_keys)} (layer4/fc, ignored)")
            if n_missing:
                raise RuntimeError(f"[DDAD-FE] {n_missing} backbone params not loaded "
                                   f"-- name mismatch? first few: {res.missing_keys[:5]}")
        else:
            print("  [DDAD-FE] WARNING: DDAD_FE_WEIGHTS not set -> ImageNet weights "
                  "(this is the pretrained/feat0 baseline)")


evaluate.ResNetPixelEmbedder = DDADFinetunedEmbedder


if __name__ == "__main__":
    evaluate.main()
