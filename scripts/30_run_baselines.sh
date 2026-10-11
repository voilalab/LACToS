#!/bin/bash
# Stage 3: baseline inference at the Table 1 settings -> $OUT/baseline_maps/<bench>/<method>/<setting>/<set>__<id>.npz
#
#   GPU=0 bash scripts/30_run_baselines.sh                # all three benchmarks (~1.5 GPU-hours, DDAD dominates)
#   GPU=0 bash scripts/30_run_baselines.sh ct             # one of: ffhq | ct | xray
#   LIMIT=2 GPU=0 bash scripts/30_run_baselines.sh ffhq   # smoke test: first two images of every set
#
# Each method runs under its own conda environment (environment/*.txt):
#   patchcore -> env "patchcore" (python 3.8, faiss)      simplenet, glass -> env "simplenet" (python 3.8)
#   supersimplenet -> env "ssn_env"                        efficientad, ddad -> the current interpreter (LACToS env)
# Override the interpreters with PY_PATCHCORE=..., PY_SIMPLENET=..., PY_SSN=..., PY_DDAD=... or the
# environment directory with CONDA_ENVS=/path/to/conda/envs. Upstream code: scripts/fetch_third_party.sh.
#
# Settings (paper Table 6 + each method's published blur, composed in stage 4):
#   PatchCore k=1 | EfficientAD alpha: CT 0 (no-pad), FFHQ 0.7, X-ray 0.2 (no-pad) | DDAD T=250 skip=25 eta=1,
#   w: CT 2, FFHQ 2, X-ray 4 (v applied in stage 4: CT 1, FFHQ 4, X-ray 4) | SimpleNet, SuperSimpleNet, GLASS: default.
set -euo pipefail
source "$(dirname "$0")/common.sh"
CONDA_ENVS="${CONDA_ENVS:-$(conda info --base 2>/dev/null)/envs}"
PY_PATCHCORE="${PY_PATCHCORE:-$CONDA_ENVS/patchcore/bin/python}"
PY_SIMPLENET="${PY_SIMPLENET:-$CONDA_ENVS/simplenet/bin/python}"
PY_SSN="${PY_SSN:-$CONDA_ENVS/ssn_env/bin/python}"
PY_DDAD="${PY_DDAD:-$(command -v python)}"           # EfficientAD and DDAD run in the LACToS environment (the current interpreter)
for py in "$PY_PATCHCORE" "$PY_SIMPLENET" "$PY_SSN" "$PY_DDAD"; do
  [ -x "$py" ] || { echo "interpreter not found: $py (see environment/ and the header of this script)"; exit 2; }
done
for d in patchcore-inspection SimpleNet SuperSimpleNet GLASS; do
  [ -d "third_party/$d" ] || { echo "third_party/$d missing: run bash scripts/fetch_third_party.sh"; exit 2; }
done
export ASSETS OUT
B="$OUT/baseline_maps"; LOG="$OUT/logs"; mkdir -p "$B" "$LOG"
LIM=""; [ -n "${LIMIT:-}" ] && LIM="--limit $LIMIT"
run() {  # python method bench [extra]
  local py=$1 m=$2 b=$3; shift 3
  echo "[$(date '+%F %T')] $b/$m"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 "$py" baselines/infer_feature.py --method $m --bench $b --out_root "$B" $LIM "$@" > "$LOG/baseline_${b}_$m.log" 2>&1 || { echo "  FAILED, see $LOG/baseline_${b}_$m.log"; exit 1; }
}
run_bench() {
  local b=$1 pad="" w
  case $b in ct) w=2; pad="--nopad";; ffhq) w=2;; xray) w=4; pad="--nopad";; *) echo "unknown benchmark '$b' (ffhq | ct | xray)"; exit 1;; esac
  run "$PY_PATCHCORE" patchcore $b
  run "$PY_SIMPLENET" simplenet $b
  run "$PY_SIMPLENET" glass $b
  run "$PY_SSN" supersimplenet $b
  run "$PY_DDAD" efficientad $b $pad
  echo "[$(date '+%F %T')] $b/ddad (w=$w, T=250, skip=25, eta=1)"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 "$PY_DDAD" baselines/infer_ddad.py --bench $b --w $w --T 250 --skip 25 --eta 1 --out_root "$B" $LIM > "$LOG/baseline_${b}_ddad.log" 2>&1 || { echo "  FAILED, see $LOG/baseline_${b}_ddad.log"; exit 1; }
}
BENCHES=("$@"); [ ${#BENCHES[@]} -eq 0 ] && BENCHES=(ffhq ct xray)
for b in "${BENCHES[@]}"; do run_bench $b; done
echo "done -> $B"
