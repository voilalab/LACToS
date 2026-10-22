#!/bin/bash
# Stage 2a: LACToS scoring at the paper's operating points (Table 5 of the paper).
# Reads the reconstructions and checkpoints from the bundle, writes one delta-map
# dump per test image:   $OUT/lactos_maps/<bench>/<tag>/<set>/samples_<id>.npz
#
#   GPU=0 bash scripts/score_lactos.sh              # all three benchmarks (~2-3 GPU-hours)
#   GPU=0 bash scripts/score_lactos.sh CT           # one of: CT | xray | ffhq
#   MAX_SAMPLES=2 GPU=0 bash scripts/score_lactos.sh ffhq   # smoke test on two images per set
#   RECON_ROOT=results/recons_mine GPU=0 bash scripts/score_lactos.sh CT   # score reconstructions you sampled yourself (stage 1)
#
# The scoring settings come from configs/lactos_*.yaml (superpixel size S, draws M, histogram sigma H,
# bins intensity x feature, feature term on/off); this script only sets the data paths of the second
# set of each benchmark and the output directories:
#   CHAOS  S=100 M=5  H=0.5 16x64 lambda=1     -> one run per set (cta, ctn), full score
#   DVXray S=10  M=20 H=0   16x16 lambda=0     -> one run per set (xrsc, xrgd), intensity-only
#   FFHQ   S=20  M=20 H=1   16x32 lambda=0.25  -> two runs per set (full + --no_pca); mixed in stage 3
# M is fixed by the shipped reconstructions (5 CT draws, 20 X-ray / FFHQ draws).
set -euo pipefail
source "$(dirname "$0")/common.sh"
M="$OUT/lactos_maps"; J="$OUT/lactos_json"; F="$OUT/figures"; LOG="$OUT/logs"; mkdir -p "$LOG"
RC="${RECON_ROOT:-assets/recons}"   # RECON_ROOT=results/recons_mine scores your own stage-1 output
SEEDED="python tools/seeded_run.py"
MS=""; [ -n "$MAX_SAMPLES" ] && MS="--max_samples $MAX_SAMPLES"

run_CT() {
  for sub in cta ctn; do
    echo "[$(date '+%F %T')] CT/$sub"
    CUDA_VISIBLE_DEVICES=$GPU $SEEDED evaluate.py --config configs/lactos_ct.yaml --subcategory $sub \
      --results_dir "$RC/results_patches_ddad_native_CT_$sub" --image_dir "assets/data/CT/test/$sub" --figures_dir "$F/CT" \
      $MS --dump_maps "$M/CT/sp100_m5_hs0.5_b16x64/$sub" --output_dir "$J/CT/$sub" > "$LOG/lactos_CT_$sub.log" 2>&1 || { echo "  FAILED, see $LOG/lactos_CT_$sub.log"; exit 1; }
  done
}
run_xray() {
  for sub in xrsc xrgd; do
    echo "[$(date '+%F %T')] xray/$sub"
    DDAD_FE_WEIGHTS="assets/checkpoints/lactos/backbone_xray_ddad_lastlayer_feat4.pth" CUDA_VISIBLE_DEVICES=$GPU \
    $SEEDED embedder/evaluate_ddad_fe.py --config configs/lactos_xray.yaml --subcategory $sub \
      --results_dir "$RC/results_patches_ddad_native_xray_$sub" --gt_root assets/data/xray/ground_truth --image_dir "assets/data/xray/test/$sub" --figures_dir "$F/xray" \
      $MS --dump_maps "$M/xray/sp10_m20_hs0_rgbonly_b16/$sub" --output_dir "$J/xray/$sub" > "$LOG/lactos_xray_$sub.log" 2>&1 || { echo "  FAILED, see $LOG/lactos_xray_$sub.log"; exit 1; }
  done
}
run_ffhq() {
  for set in faces moreface faces_normal; do
    for variant in full rgbonly; do
      tag=sp20_m20_hs1_b16x32; extra=""
      if [ $variant = rgbonly ]; then tag=sp20_m20_hs1_rgbonly_b16; extra="--no_pca"; fi
      echo "[$(date '+%F %T')] ffhq/$set ($variant)"
      CUDA_VISIBLE_DEVICES=$GPU $SEEDED evaluate.py --config configs/lactos_ffhq_$set.yaml \
        --results_dir "$RC/results_patches_adm_dps_$set" --figures_dir "$F/ffhq_$set" $extra \
        $MS --dump_maps "$M/ffhq/$tag/$set" --output_dir "$J/ffhq/$set" > "$LOG/lactos_ffhq_${set}_$variant.log" 2>&1 || { echo "  FAILED, see $LOG/lactos_ffhq_${set}_$variant.log"; exit 1; }
    done
  done
}

BENCHES=("$@"); [ ${#BENCHES[@]} -eq 0 ] && BENCHES=(CT xray ffhq)
for b in "${BENCHES[@]}"; do
  case $b in CT|xray|ffhq) run_$b ;; *) echo "unknown benchmark '$b' (CT | xray | ffhq)"; exit 1 ;; esac
done
echo "done -> $M"
