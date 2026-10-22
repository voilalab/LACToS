#!/bin/bash
# Stage 0: check the environment and that the extracted asset bundle is complete
# (required files, image and reconstruction counts, reference results).
#   bash scripts/check_assets.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
python - <<'PY'
import sys
try:
    import torch, torchvision, skimage, sklearn, scipy
except ImportError as e:
    sys.exit(f"missing package: {e} (pip install -r requirements.txt)")
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  torchvision {torchvision.__version__}  "
      f"scikit-image {skimage.__version__}  scikit-learn {sklearn.__version__}  scipy {scipy.__version__}  "
      f"cuda {'available' if torch.cuda.is_available() else 'NOT available'}")
if not torch.__version__.startswith("2.0.1"):
    print(f"WARNING: torch {torch.__version__} is installed; the paper's numbers were produced with torch 2.0.1 "
          "(pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118)")
PY
python tools/check_bundle.py "$ASSETS"
