#!/bin/bash
# Stage 0: verify the extracted asset bundle (layout, counts, SHA-256).
#   bash scripts/00_check_assets.sh            # full check incl. checksums of every file
#   bash scripts/00_check_assets.sh --no-sha   # layout and counts only (seconds)
set -euo pipefail
source "$(dirname "$0")/common.sh"
python tools/check_bundle.py "$ASSETS" "$@"
