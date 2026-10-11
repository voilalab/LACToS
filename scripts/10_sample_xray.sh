#!/bin/bash
# Stage 1 (optional): regenerate the DVXray reconstructions with DPS.
# The bundle already ships them (assets/recons/results_patches_ddad_native_xray_{xrsc,xrgd}).
#
#   GPU=0 bash scripts/10_sample_xray.sh                      # both sets, 72 images each, 20 draws
#   GPU=0 SHARD=0 NSHARDS=4 bash scripts/10_sample_xray.sh    # one of 4 shards
#   GPU=0 IMAGES=P01809_OL DRAWS=1 bash scripts/10_sample_xray.sh xrsc   # smoke test
#
# Paper setting (Table 5): y = x + sigma n with sigma = 2.0 (images in [-1, 1]), DPS scale 1.0
# (zeta' = 0.5), 1000 DDPM steps, 20 draws, DVXray guided-diffusion prior trained on the 5,000
# weapon-free bags (assets/checkpoints/diffusion/xray_model500000.pt).
set -euo pipefail
source "$(dirname "$0")/common.sh"
DRAWS="${DRAWS:-20}"; SHARD="${SHARD:-0}"; NSHARDS="${NSHARDS:-1}"
RAW="$OUT/recons_mine/raw"; mkdir -p "$RAW" "$OUT/logs"
IMG=""; [ -n "${IMAGES:-}" ] && IMG="--images $IMAGES"
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(xrsc xrgd)
for sub in "${SETS[@]}"; do
  echo "[$(date '+%F %T')] xray/$sub -> $RAW/xray_$sub (shard $SHARD/$NSHARDS)"
  ( cd sampling/dps && CUDA_VISIBLE_DEVICES=$GPU python sample_fast.py \
      --model_config configs/xray_model_config.yaml --model_path "$ROOT/assets/checkpoints/diffusion/xray_model500000.pt" \
      --diffusion_config configs/diffusion_config.yaml --task_config configs/xray_inpainting_sigma2.yaml \
      --data_root "$ROOT/assets/data/xray/test/$sub" --save_dir "$RAW/xray_$sub" \
      --num_measurements "$DRAWS" --batch 14 --scale 1.0 --grad_norm per_sample --seed 3 \
      --shard "$SHARD" --num_shards "$NSHARDS" $IMG ) 2>&1 | tee "$OUT/logs/sample_xray_${sub}_shard$SHARD.log" | grep -v "^\[.*done |" || true
  python sampling/wire_recons.py --src "$RAW/xray_$sub" --dst "$OUT/recons_mine/results_patches_ddad_native_xray_$sub" --n_draws "$DRAWS"
done
echo "done. Score them with:  RECON_ROOT=$OUT/recons_mine GPU=$GPU bash scripts/20_score_lactos.sh xray"
