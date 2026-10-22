#!/bin/bash
# Train every missing toy prior, one per GPU at a time (~8.5 h each on an RTX 6000 Ada):
# quad_L1..L4 (toy 1) and quadp_p40..p10 (toy 2; quadp_p50 reuses quad_L1).
#   GPUS="0 1 2 3 4 5 6 7" bash toy/train_priors.sh            # all eight in parallel
#   GPUS="0 1" bash toy/train_priors.sh quad_L1 quad_L2        # a subset
set -euo pipefail
source "$(dirname "$0")/common.sh"
DS=("$@"); [ ${#DS[@]} -eq 0 ] && DS=(quad_L1 quad_L2 quad_L3 quad_L4 quadp_p40 quadp_p30 quadp_p20 quadp_p10)
LOG="$OUT/logs/train"; mkdir -p "$LOG"
for d in "${DS[@]}"; do
  [ -d "$DATA/$d/train/good" ] || { echo "no data for $d: run python toy/gen_data.py"; exit 2; }
  if [ -f "$CKPT/$d/$EPOCH" ]; then say "$d: $CKPT/$d/$EPOCH exists, skipping"; continue; fi
  echo "train_$d|python toy/train_prior.py --dataset $d --data_root $DATA --out $CKPT"
done | run_jobs "$LOG"
say "done -> $CKPT"
