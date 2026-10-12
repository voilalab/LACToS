#!/bin/bash
# Stage 1b: assemble an asset bundle whose recons/ are the reconstructions sampled in stage 1.
# This is the directory that is archived and uploaded (README, "What is in the asset bundle").
#
#   bash scripts/15_pack_bundle.sh /path/to/LACToS_table1_assets_new
#
# checkpoints/ and data/ are copied from the current bundle ($ASSETS). Each of the seven
# reconstruction sets is taken from $OUT/recons_mine if it was sampled there, otherwise from the
# current bundle (the source of every set is printed). SHA256SUMS is written and the result is
# checked. The current bundle is never modified.
set -euo pipefail
source "$(dirname "$0")/common.sh"
NEW="${1:?usage: bash scripts/15_pack_bundle.sh /path/to/new_bundle_dir}"
case "$NEW" in /*) ;; *) NEW="$PWD/$NEW" ;; esac
[ -e "$NEW" ] && { echo "$NEW already exists; choose a new directory"; exit 2; }
MINE="$OUT/recons_mine"
mkdir -p "$NEW/recons"
echo "copying checkpoints/ and data/ from $ASSETS ..."
cp -a "$ASSETS/checkpoints" "$ASSETS/data" "$NEW/"
for d in results_patches_ddad_native_CT_cta results_patches_ddad_native_CT_ctn \
         results_patches_ddad_native_xray_xrsc results_patches_ddad_native_xray_xrgd \
         results_patches_adm_dps_faces results_patches_adm_dps_moreface results_patches_adm_dps_faces_normal; do
  if [ -d "$MINE/$d" ]; then src="$MINE/$d"; tag="sampled here"; else src="$ASSETS/recons/$d"; tag="from the current bundle"; fi
  [ -d "$src" ] || { echo "  $d: not found in $MINE or $ASSETS/recons"; exit 1; }
  cp -a "$src" "$NEW/recons/$d"
  printf "  %-44s %s\n" "$d" "$tag"
done
python tools/check_bundle.py "$NEW" --write-sha
B="$(basename "$NEW")"
cat <<MSG

bundle ready: $NEW
  archive it:   tar -cf $B.tar -C "$(dirname "$NEW")" "$B" && sha256sum $B.tar
  use it here:  rm -f "$ROOT/assets" && ln -s "$NEW" "$ROOT/assets"
MSG
