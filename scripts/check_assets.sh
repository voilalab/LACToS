#!/bin/bash
# Stage 0: check that the extracted asset bundle is complete (required files, image and reconstruction counts).
#   bash scripts/check_assets.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
python tools/check_bundle.py "$ASSETS"
