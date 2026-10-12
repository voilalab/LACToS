#!/bin/bash
# Stage 1 (DVXray): sample the DPS reconstructions of the X-ray test images.
# 144 images (xrsc = scissors, xrgd = weapon-free) x 20 draws = 2,880 reconstructions.
#
#   GPUS="0 1 2 3" bash scripts/10_sample_xray.sh                        # both sets, one shard per listed GPU
#   GPU=0 bash scripts/10_sample_xray.sh xrsc                            # one set on one GPU
#   GPU=0 IMAGES=P01809_OL DRAWS=1 bash scripts/10_sample_xray.sh xrsc   # smoke test
#
# Re-running resumes: reconstructions already on disk are skipped.
# Output: $OUT/recons_mine/results_patches_ddad_native_xray_{xrsc,xrgd}/   (the layout of recons/ in the asset bundle)
#
# Setting (paper Table 5): y = x + sigma n with sigma = 2.0 (images in [-1, 1]), DPS scale 1.0 (zeta' = 0.5),
# 1000 DDPM steps, 20 draws, DVXray guided-diffusion prior (assets/checkpoints/diffusion/xray_model500000.pt).
# As for CHAOS, every run gives new posterior draws (sampling/README.md, "Determinism").
set -euo pipefail
source "$(dirname "$0")/common.sh"
DRAWS="${DRAWS:-20}"
read -r -a GPU_LIST <<< "${GPUS:-$GPU}"; N=${#GPU_LIST[@]}
RAW="$OUT/recons_mine/raw"; LOG="$OUT/logs"; mkdir -p "$RAW" "$LOG"
IMG=""; [ -n "${IMAGES:-}" ] && IMG="--images $IMAGES"
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(xrsc xrgd)
for sub in "${SETS[@]}"; do
  echo "[$(date '+%F %T')] xray/$sub on GPU(s) ${GPU_LIST[*]}  (progress: tail -f $LOG/sample_xray_${sub}_shard*.log)"
  pids=()
  for i in "${!GPU_LIST[@]}"; do
    ( cd sampling/dps && CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python sample_fast.py \
        --model_config configs/xray_model_config.yaml --model_path "$ROOT/assets/checkpoints/diffusion/xray_model500000.pt" \
        --diffusion_config configs/diffusion_config.yaml --task_config configs/xray_inpainting_sigma2.yaml \
        --data_root "$ROOT/assets/data/xray/test/$sub" --save_dir "$RAW/xray_$sub" \
        --num_measurements "$DRAWS" --batch 14 --scale 1.0 --grad_norm per_sample --seed 3 \
        --start 0 --end 99999 --shard "$i" --num_shards "$N" $IMG ) > "$LOG/sample_xray_${sub}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || { echo "  sampling FAILED, see $LOG/sample_xray_${sub}_shard*.log"; exit 1; }
  python sampling/wire_recons.py --src "$RAW/xray_$sub" --dst "$OUT/recons_mine/results_patches_ddad_native_xray_$sub" --n_draws "$DRAWS"
done
echo "done -> $OUT/recons_mine   (score: RECON_ROOT=$OUT/recons_mine GPU=${GPU_LIST[0]} bash scripts/20_score_lactos.sh xray)"
