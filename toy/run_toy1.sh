#!/bin/bash
# Toy experiment 1 (paper Figure 2, Appendix Figure 7): false positives vs. inlier vocabulary.
# Needs the datasets (toy/gen_data.py), the priors quad_L1..L4 (toy/train_priors.sh), the
# patchcore environment and third_party/patchcore-inspection (scripts/fetch_third_party.sh).
#
#   GPUS="0 1 2 3" bash toy/run_toy1.sh            # ~2 h on four GPUs; resumable
#   GPUS="0" LEVELS="1" NGOOD=2 NANOM=2 bash toy/run_toy1.sh    # smoke test
#
# Per level: DPS reconstructions (sigma 0.30, 20 seeds) of the 50 anomalous and the first 50
# anomaly-free test images; SLIC masks; DDAD maps (1 sample, 20 sample mean); LACToS maps with
# 1, 5, 10, 20 samples; PatchCore maps (one memory bank per level); then toy/analyze_toy1.py.
# Output: $OUT/quad/L<N>/... and $OUT/quad/toy1_fpr_additive_dps_s030.{json,md,png}
set -euo pipefail
source "$(dirname "$0")/common.sh"
SIGMA="${SIGMA:-0.30}"; CELL="additive_dps_s$SIGMA"; NSEEDS="${NSEEDS:-20}"; KS=(${KS:-1 5 10 20})
LEVELS=(${LEVELS:-1 2 3 4}); NGOOD="${NGOOD:-50}"; NANOM="${NANOM:-100000}"
PY_PATCHCORE="${PY_PATCHCORE:-$(conda info --base 2>/dev/null)/envs/patchcore/bin/python}"
R="$OUT/quad"; LOG="$OUT/logs/toy1"; mkdir -p "$R" "$LOG"
img_of() { echo "$DATA/quad_L$1/test/$2"; }
base_of() { case "$2" in good) echo "$R/L$1/good" ;; *) echo "$R/L$1" ;; esac; }
n_of() { case "$2" in good) echo "$NGOOD" ;; *) echo "$NANOM" ;; esac; }
for L in "${LEVELS[@]}"; do [ -d "$(img_of $L q1_square)" ] || { echo "no data for quad_L$L: run python toy/gen_data.py"; exit 2; }; need_prior quad_L$L > /dev/null; done

say "=== 1. reconstructions ($CELL, $NSEEDS seeds; anomalous + first $NGOOD normal images per level) ==="
for L in "${LEVELS[@]}"; do for S in q1_square good; do
  rec="$(base_of $L $S)/recons/$CELL"; sids=($(sids_of "$(img_of $L $S)" "$(n_of $L $S)"))
  [ "$(count_pngs "$rec")" -ge $(( ${#sids[@]} * NSEEDS )) ] && continue
  write_recon_cfg "$rec.yaml" "$SIGMA" "$(need_prior quad_L$L)" "$(img_of $L $S)" "$rec" "$NSEEDS" "${sids[@]}"
  echo "recon_L${L}_$S|python sampling/generate_recons.py --recon_config $rec.yaml"
done; done | run_jobs "$LOG"

say "=== 2. SLIC masks ==="
for L in "${LEVELS[@]}"; do for S in q1_square good; do
  python toy/prepare_superpixels.py --image_dir "$(img_of $L $S)" --figures_root "$(base_of $L $S)/figures" \
    --sids $(sids_of "$(img_of $L $S)" "$(n_of $L $S)") > "$LOG/superpixels_L${L}_$S.log" 2>&1 &
done; done; wait

say "=== 3. scoring (DDAD single + mean of $NSEEDS, LACToS set of ${KS[*]}) ==="
for L in "${LEVELS[@]}"; do for S in q1_square good; do
  rec="$(base_of $L $S)/recons/$CELL"; img="$(img_of $L $S)"; sids=$(sids_of "$img" "$(n_of $L $S)"); n=$(echo "$sids" | wc -w)
  o="$(base_of $L $S)/$CELL/single/maps/ddad"; [ "$(count_maps "$o")" -ge "$n" ] || echo "score_L${L}_${S}_ddad_single|python toy/score_from_recons.py --method ddad --mode single --recons $rec --image_dir $img --out_dir $o --sids $sids"
  o="$(base_of $L $S)/$CELL/mean$NSEEDS/maps/ddad"; [ "$(count_maps "$o")" -ge "$n" ] || echo "score_L${L}_${S}_ddad_mean|python toy/score_from_recons.py --method ddad --mode mean --k $NSEEDS --recons $rec --image_dir $img --out_dir $o --sids $sids"
  for K in "${KS[@]}"; do
    o="$(base_of $L $S)/$CELL/set$K/maps/lactos"; [ "$(count_maps "$o")" -ge "$n" ] || echo "score_L${L}_${S}_lactos_set$K|python toy/score_from_recons.py --method lactos --mode set --k $K --recons $rec --image_dir $img --out_dir $o --sids $sids --figures_root $(base_of $L $S)/figures"
  done
done; done | run_jobs "$LOG"

say "=== 4. PatchCore ($PY_PATCHCORE) ==="
[ -x "$PY_PATCHCORE" ] || { echo "interpreter not found: $PY_PATCHCORE (environment/patchcore.txt; set PY_PATCHCORE=...)"; exit 2; }
for L in "${LEVELS[@]}"; do
  o="$R/L$L/patchcore"; [ "$(count_maps "$o/good/maps/patchcore")" -ge "$NGOOD" ] && [ "$(count_maps "$o/q1_square/maps/patchcore")" -ge 1 ] && continue
  echo "patchcore_L$L|$PY_PATCHCORE toy/run_patchcore.py --data_root $DATA/quad_L$L --test_subdirs q1_square good --n_test $NGOOD --out_root $o ${MAX_TRAIN:+--max_train $MAX_TRAIN}"
done | run_jobs "$LOG"

say "=== 5. analysis ==="
python toy/analyze_toy1.py --results_root "$R" --data_root "$DATA" --cell "$CELL"
say "done -> $R"
