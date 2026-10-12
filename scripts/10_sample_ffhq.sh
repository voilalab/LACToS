#!/bin/bash
# Stage 1 (FFHQ): sample the DPS reconstructions of the face test images with the guided-diffusion FFHQ model.
# 100 images (faces 22, moreface 28, faces_normal 50) x 20 seeds = 2,000 reconstructions.
#
#   GPUS="0 1 2 3" bash scripts/10_sample_ffhq.sh                     # all three sets, images split over the listed GPUs
#   GPU=0 bash scripts/10_sample_ffhq.sh faces_normal                 # one set on one GPU
#   GPU=0 SAMPLES="40" SEEDS=1 bash scripts/10_sample_ffhq.sh faces   # smoke test
#
# Re-running resumes: images whose seeds are all on disk are skipped.
# Output: $OUT/recons_mine/results_patches_adm_dps_{faces,moreface,faces_normal}/   (the layout of recons/ in the asset bundle)
#
# Setting (paper Table 5): y = x + sigma n with sigma = 0.01, DPS scale 0.1, 1000 steps (skip 1), eta 1,
# seeds 0-19, checkpoint ffhq_10m.pt (assets/checkpoints/diffusion/). This sampler is seeded per draw:
# a re-run reproduces the reconstructions in the bundle up to PNG rounding.
set -euo pipefail
source "$(dirname "$0")/common.sh"
SEEDS="${SEEDS:-20}"
read -r -a GPU_LIST <<< "${GPUS:-$GPU}"; N=${#GPU_LIST[@]}
LOG="$OUT/logs"; mkdir -p "$OUT/recons_mine" "$LOG"
SMP=""; [ -n "${SAMPLES:-}" ] && SMP="--samples $SAMPLES"
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(faces moreface faces_normal)
for set in "${SETS[@]}"; do
  echo "[$(date '+%F %T')] ffhq/$set on GPU(s) ${GPU_LIST[*]}  (progress: tail -f $LOG/sample_ffhq_${set}_shard*.log)"
  pids=()
  for i in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python sampling/generate_recons.py --recon_config "configs/recon/adm_dps_ffhq_$set.yaml" \
      --ckpt assets/checkpoints/diffusion/ffhq_10m.pt --image_dir "assets/data/ffhq/$set/test" \
      --out_root "$OUT/recons_mine/results_patches_adm_dps_$set" --num_seeds "$SEEDS" \
      --shard "$i" --num_shards "$N" $SMP > "$LOG/sample_ffhq_${set}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || { echo "  sampling FAILED, see $LOG/sample_ffhq_${set}_shard*.log"; exit 1; }
done
echo "done -> $OUT/recons_mine   (score: RECON_ROOT=$OUT/recons_mine GPU=${GPU_LIST[0]} bash scripts/20_score_lactos.sh ffhq)"
