#!/bin/bash
# Stage 1 (CHAOS): sample the DPS reconstructions of the CT test slices.
# 200 slices (cta = with tumour, ctn = normal) x 5 draws = 1,000 reconstructions.
#
#   GPUS="0 1 2 3" bash scripts/10_sample_ct.sh                        # both sets, one shard per listed GPU
#   GPU=0 bash scripts/10_sample_ct.sh cta                             # one set on one GPU
#   GPU=0 IMAGES=image_0000 DRAWS=1 bash scripts/10_sample_ct.sh cta   # smoke test
#
# Re-running resumes: reconstructions already on disk are skipped.
# Output: $OUT/recons_mine/results_patches_ddad_native_CT_{cta,ctn}/   (the layout of recons/ in the asset bundle)
#         $OUT/recons_mine/raw/CT_{cta,ctn}/                           (sampler output before re-layout)
#
# Setting (paper Table 5): identity forward model y = x + sigma n with sigma = 0.1, DPS scale 0.4
# (zeta' = 0.2 in the normalised step), 1000 DDPM steps, 5 draws, CHAOS guided-diffusion prior
# (assets/checkpoints/diffusion/ct_model050000.pt). The sampler seeds the measurement noise and x_T
# per (image, draw) but not the reverse-step noise, so every run gives new posterior draws
# (sampling/README.md, "Determinism").
set -euo pipefail
source "$(dirname "$0")/common.sh"
DRAWS="${DRAWS:-5}"
read -r -a GPU_LIST <<< "${GPUS:-$GPU}"; N=${#GPU_LIST[@]}
RAW="$OUT/recons_mine/raw"; LOG="$OUT/logs"; mkdir -p "$RAW" "$LOG"
IMG=""; [ -n "${IMAGES:-}" ] && IMG="--images $IMAGES"
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(cta ctn)
for sub in "${SETS[@]}"; do
  echo "[$(date '+%F %T')] CT/$sub on GPU(s) ${GPU_LIST[*]}  (progress: tail -f $LOG/sample_CT_${sub}_shard*.log)"
  pids=()
  for i in "${!GPU_LIST[@]}"; do
    ( cd sampling/dps && CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python sample_fast.py \
        --model_config configs/ct_model_config.yaml --model_path "$ROOT/assets/checkpoints/diffusion/ct_model050000.pt" \
        --diffusion_config configs/diffusion_config.yaml --task_config configs/ctgrid/sigma0.1.yaml \
        --data_root "$ROOT/assets/data/CT/test/$sub" --save_dir "$RAW/CT_$sub" \
        --num_measurements "$DRAWS" --batch 16 --scale 0.4 --grad_norm per_sample --seed 3 \
        --start 0 --end 99999 --shard "$i" --num_shards "$N" $IMG ) > "$LOG/sample_CT_${sub}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || { echo "  sampling FAILED, see $LOG/sample_CT_${sub}_shard*.log"; exit 1; }
  # the ctn images are already named image_6XXX in the bundle, so the reconstructions get those ids directly
  python sampling/wire_recons.py --src "$RAW/CT_$sub" --dst "$OUT/recons_mine/results_patches_ddad_native_CT_$sub" --n_draws "$DRAWS"
done
echo "done -> $OUT/recons_mine   (score: RECON_ROOT=$OUT/recons_mine GPU=${GPU_LIST[0]} bash scripts/20_score_lactos.sh CT)"
