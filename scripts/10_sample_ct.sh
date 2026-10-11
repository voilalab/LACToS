#!/bin/bash
# Stage 1 (optional): regenerate the CHAOS CT reconstructions with DPS.
# The bundle already ships them (assets/recons/results_patches_ddad_native_CT_{cta,ctn});
# run this only to re-sample. Output goes to $OUT/recons_mine/ and never touches the bundle.
#
#   GPU=0 bash scripts/10_sample_ct.sh                       # both sets, all 100 slices each, 5 draws
#   GPU=0 SHARD=0 NSHARDS=4 bash scripts/10_sample_ct.sh     # one of 4 shards (run 4 of these on 4 GPUs; draws are per-image seeded)
#   GPU=0 IMAGES=image_0000 DRAWS=1 bash scripts/10_sample_ct.sh cta   # smoke test
#
# Paper setting (Table 5): identity forward model y = x + sigma n, sigma = 0.1, DPS scale 0.4
# (zeta' = 0.2 in the normalised step), 1000 DDPM steps, CT guided-diffusion prior
# (assets/checkpoints/diffusion/ct_model050000.pt). The paper's M = 5 draws are the first
# 5 of a 10-draw run; draws are seeded per (image, draw index), so --num_measurements 5
# reproduces them.
set -euo pipefail
source "$(dirname "$0")/common.sh"
DRAWS="${DRAWS:-5}"; SHARD="${SHARD:-0}"; NSHARDS="${NSHARDS:-1}"
RAW="$OUT/recons_mine/raw"; mkdir -p "$RAW" "$OUT/logs"
IMG=""; [ -n "${IMAGES:-}" ] && IMG="--images $IMAGES"
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(cta ctn)
for sub in "${SETS[@]}"; do
  echo "[$(date '+%F %T')] CT/$sub -> $RAW/CT_$sub (shard $SHARD/$NSHARDS)"
  ( cd sampling/dps && CUDA_VISIBLE_DEVICES=$GPU python sample_fast.py \
      --model_config configs/ct_model_config.yaml --model_path "$ROOT/assets/checkpoints/diffusion/ct_model050000.pt" \
      --diffusion_config configs/diffusion_config.yaml --task_config configs/ctgrid/sigma0.1.yaml \
      --data_root "$ROOT/assets/data/CT/test/$sub" --save_dir "$RAW/CT_$sub" \
      --num_measurements "$DRAWS" --batch 16 --scale 0.4 --grad_norm per_sample --seed 3 \
      --shard "$SHARD" --num_shards "$NSHARDS" $IMG ) 2>&1 | tee "$OUT/logs/sample_CT_${sub}_shard$SHARD.log" | grep -v "^\[.*done |" || true
  # the ctn images are already named image_6XXX in the bundle, so no id offset is needed
  python sampling/wire_recons.py --src "$RAW/CT_$sub" --dst "$OUT/recons_mine/results_patches_ddad_native_CT_$sub" --n_draws "$DRAWS"
done
echo "done. Score them with:  RECON_ROOT=$OUT/recons_mine GPU=$GPU bash scripts/20_score_lactos.sh CT"
