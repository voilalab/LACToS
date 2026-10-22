#!/bin/bash
# Toy experiment 2 (paper Appendix Figure 9): AUROC against the square probability p.
# Needs the datasets (toy/gen_data.py) and the priors quad_L1 (for quadp_p50) and quadp_p40..p10
# (toy/train_priors.sh).
#
#   GPUS="0 1 2 3" bash toy/run_toy2.sh            # ~3 h on four GPUs; resumable
#   GPUS="0" DATASETS="quadp_p50" NANOM=2 NSEEDS=2 KS="1 2" bash toy/run_toy2.sh    # smoke test
#
# Per prior: DPS reconstructions (sigma 0.10, 40 seeds) of the 50 shared test images; SLIC masks;
# DDAD maps (1 sample, 20 sample mean); LACToS maps with 1, 2, 5, 10, 20, 40 samples; then
# toy/analyze_toy2.py. Output: $OUT/quadp/<dataset>/... and $OUT/quadp/toy2_auroc_additive_dps_s010.{json,md,png}
set -euo pipefail
source "$(dirname "$0")/common.sh"
SIGMA="${SIGMA:-0.10}"; CELL="additive_dps_s$SIGMA"; NSEEDS="${NSEEDS:-40}"; KS=(${KS:-1 2 5 10 20 40}); NMEAN="${NMEAN:-20}"
DATASETS=(${DATASETS:-quadp_p50 quadp_p40 quadp_p30 quadp_p20 quadp_p10}); NANOM="${NANOM:-100000}"; V=q1_square_plus
R="$OUT/quadp"; LOG="$OUT/logs/toy2"; mkdir -p "$R" "$LOG"
for d in "${DATASETS[@]}"; do [ -d "$DATA/$d/test/$V" ] || { echo "no data for $d: run python toy/gen_data.py --p_sweep"; exit 2; }; need_prior $d > /dev/null; done

say "=== 1. reconstructions ($CELL, $NSEEDS seeds) ==="
for d in "${DATASETS[@]}"; do
  rec="$R/$d/recons/$CELL"; img="$DATA/$d/test/$V"; sids=($(sids_of "$img" "$NANOM"))
  [ "$(count_pngs "$rec")" -ge $(( ${#sids[@]} * NSEEDS )) ] && continue
  write_recon_cfg "$rec.yaml" "$SIGMA" "$(need_prior $d)" "$img" "$rec" "$NSEEDS" "${sids[@]}"
  echo "recon_$d|python sampling/generate_recons.py --recon_config $rec.yaml"
done | run_jobs "$LOG"

say "=== 2. SLIC masks ==="
for d in "${DATASETS[@]}"; do
  python toy/prepare_superpixels.py --image_dir "$DATA/$d/test/$V" --figures_root "$R/$d/figures" --sids $(sids_of "$DATA/$d/test/$V" "$NANOM") > "$LOG/superpixels_$d.log" 2>&1 &
done; wait

say "=== 3. scoring (DDAD single + mean of $NMEAN, LACToS set of ${KS[*]}) ==="
for d in "${DATASETS[@]}"; do
  rec="$R/$d/recons/$CELL"; img="$DATA/$d/test/$V"; sids=$(sids_of "$img" "$NANOM"); n=$(echo "$sids" | wc -w)
  o="$R/$d/$CELL/single/maps/ddad"; [ "$(count_maps "$o")" -ge "$n" ] || echo "score_${d}_ddad_single|python toy/score_from_recons.py --method ddad --mode single --recons $rec --image_dir $img --out_dir $o --sids $sids"
  o="$R/$d/$CELL/mean$NMEAN/maps/ddad"; [ "$(count_maps "$o")" -ge "$n" ] || echo "score_${d}_ddad_mean|python toy/score_from_recons.py --method ddad --mode mean --k $NMEAN --recons $rec --image_dir $img --out_dir $o --sids $sids"
  for K in "${KS[@]}"; do
    o="$R/$d/$CELL/set$K/maps/lactos"; [ "$(count_maps "$o")" -ge "$n" ] || echo "score_${d}_lactos_set$K|python toy/score_from_recons.py --method lactos --mode set --k $K --recons $rec --image_dir $img --out_dir $o --sids $sids --figures_root $R/$d/figures"
  done
done | run_jobs "$LOG"

say "=== 4. analysis ==="
python toy/analyze_toy2.py --results_root "$R" --data_root "$DATA" --cell "$CELL" --ks "${KS[@]}"
say "done -> $R"
