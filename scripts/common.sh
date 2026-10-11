# Sourced by every script in scripts/. Not meant to be run directly.
#
# Environment knobs (all optional):
#   ASSETS=/path/to/bundle   extracted asset bundle; default <repo>/assets (a symlink <repo>/assets -> $ASSETS is created when ASSETS points elsewhere)
#   OUT=/path/to/results     where results are written (default <repo>/results)
#   GPU=0                    CUDA device index for GPU stages
#   MAX_SAMPLES=N            score only the first N images of every set (smoke tests)
#   ABL_SEED=0               seed for tools/seeded_run.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/results}"
GPU="${GPU:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

_fail() { echo "$*" >&2; echo "extract LACToS_table1_assets.tar and either place/symlink it at $ROOT/assets or set ASSETS=/path/to/LACToS_table1_assets" >&2; exit 2; }
if [ -n "${ASSETS:-}" ]; then
  [ -d "$ASSETS/checkpoints" ] || _fail "asset bundle not found at ASSETS=$ASSETS"
  ASSETS="$(cd "$ASSETS" && pwd -P)"
  if [ -L "$ROOT/assets" ] && [ ! -e "$ROOT/assets" ]; then rm "$ROOT/assets"; fi        # dangling link from an old checkout
  if [ -e "$ROOT/assets" ]; then
    _have="$(cd "$ROOT/assets" 2>/dev/null && pwd -P)"
    [ "$_have" = "$ASSETS" ] || _fail "$ROOT/assets already exists and points to $_have, but ASSETS=$ASSETS; remove one of the two"
  else
    ln -s "$ASSETS" "$ROOT/assets"      # configs/*.yaml refer to assets/ relative to the repository root
  fi
else
  if [ -L "$ROOT/assets" ] && [ ! -e "$ROOT/assets" ]; then
    _fail "$ROOT/assets is a dangling symlink to $(readlink "$ROOT/assets")"
  fi
  [ -d "$ROOT/assets/checkpoints" ] || _fail "asset bundle not found at $ROOT/assets"
  ASSETS="$(cd "$ROOT/assets" && pwd -P)"
fi
export ASSETS OUT
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ABL_SEED="${ABL_SEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OOD_CT_MASK_ROOT="$ROOT/assets/data/CT/masks"   # CT lesion + body masks (evaluate.py, ood/data.py)
mkdir -p "$OUT"
cd "$ROOT"
