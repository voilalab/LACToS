# Sourced by every script in scripts/. Not meant to be run directly.
#
# Environment knobs (all optional):
#   ASSETS=/path/to/bundle   extracted asset bundle (default <repo>/assets; a symlink is created if ASSETS is elsewhere)
#   OUT=/path/to/results     where results are written (default <repo>/results)
#   GPU=0                    CUDA device index for GPU stages
#   MAX_SAMPLES=N            score only the first N images of every set (smoke tests)
#   ABL_SEED=0               seed for tools/seeded_run.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${ASSETS:-$ROOT/assets}"
OUT="${OUT:-$ROOT/results}"
GPU="${GPU:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
if [ ! -d "$ASSETS/checkpoints" ]; then
  echo "asset bundle not found at $ASSETS" >&2
  echo "extract LACToS_table1_assets.tar and either place it at $ROOT/assets or set ASSETS=/path/to/LACToS_table1_assets" >&2
  exit 2
fi
# configs/*.yaml refer to assets/ relative to the repository root
if [ ! -e "$ROOT/assets" ]; then ln -s "$ASSETS" "$ROOT/assets"; fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ABL_SEED="${ABL_SEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OOD_CT_MASK_ROOT="$ROOT/assets/data/CT/masks"   # CT lesion + body masks (evaluate.py, ood/data.py)
mkdir -p "$OUT"
cd "$ROOT"
