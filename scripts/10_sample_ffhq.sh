#!/bin/bash
# Stage 1 (optional): regenerate the FFHQ reconstructions with DPS on the guided-diffusion FFHQ model.
# The bundle already ships them (assets/recons/results_patches_adm_dps_{faces,moreface,faces_normal}).
#
#   GPU=0 bash scripts/10_sample_ffhq.sh                       # all three sets (22 + 28 + 50 images x 20 seeds, ~8 GPU-hours)
#   GPU=0 bash scripts/10_sample_ffhq.sh faces_normal          # one set
#   GPU=0 SAMPLES="40" SEEDS=1 bash scripts/10_sample_ffhq.sh faces   # smoke test
#
# Paper setting (Table 5): y = x + sigma n with sigma = 0.01, DPS scale 0.1, 1000 steps (skip 1),
# eta 1, 20 seeds, DPS2022 FFHQ checkpoint ffhq_10m.pt (assets/checkpoints/diffusion/).
set -euo pipefail
source "$(dirname "$0")/common.sh"
SEEDS="${SEEDS:-20}"; mkdir -p "$OUT/recons_mine" "$OUT/logs"
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(faces moreface faces_normal)
for set in "${SETS[@]}"; do
  echo "[$(date '+%F %T')] ffhq/$set -> $OUT/recons_mine/results_patches_adm_dps_$set"
  SMP=""; [ -n "${SAMPLES:-}" ] && SMP="--samples $SAMPLES"
  CUDA_VISIBLE_DEVICES=$GPU python sampling/generate_recons.py --recon_config "configs/recon/adm_dps_ffhq_$set.yaml" \
    --ckpt assets/checkpoints/diffusion/ffhq_10m.pt --image_dir "assets/data/ffhq/$set/test" \
    --out_root "$OUT/recons_mine/results_patches_adm_dps_$set" --num_seeds "$SEEDS" $SMP 2>&1 | tee "$OUT/logs/sample_ffhq_$set.log" | grep -v "seeds .* done" || true
done
echo "done. Score them by pointing the FFHQ configs' data.results_dir at $OUT/recons_mine (see sampling/README.md)."
