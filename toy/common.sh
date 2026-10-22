# Sourced by toy/run_toy1.sh, toy/run_toy2.sh and toy/train_priors.sh. Not meant to be run directly.
#
#   GPUS="0 1 2 3"           GPUs to use, one job per GPU at a time (required for the GPU phases)
#   OUT=results/toy          where results go (default <repo>/results/toy)
#   DATA=toy/data            the datasets (toy/gen_data.py)
#   CKPT=toy/checkpoints     the priors (toy/train_prior.py), <CKPT>/<dataset>/<epoch>
#   EPOCH=250                which checkpoint of every prior to use
#   PY_PATCHCORE=...         interpreter of the `patchcore` environment (toy 1 only)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/results/toy}"; DATA="${DATA:-$ROOT/toy/data}"; CKPT="${CKPT:-$ROOT/toy/checkpoints}"; EPOCH="${EPOCH:-250}"
read -r -a GPU_LIST <<< "${GPUS:-0}"; NGPU=${#GPU_LIST[@]}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$ROOT"
say() { echo "[$(date '+%F %T')] $*"; }
count_pngs() { find "$1" -path '*inpainting/recon/*' -name '*.png' 2>/dev/null | wc -l; }
count_maps() { ls "$1"/*.npy 2>/dev/null | wc -l; }
sids_of() {  # first N image ids (sorted) of a folder, space-separated on ONE line (job lines must stay single lines)
  local -a f=("$1"/*.png) ids=(); local n="${2:-100000}" x
  for x in "${f[@]:0:$n}"; do x=${x##*/}; ids+=("${x%.png}"); done
  echo "${ids[*]}"
}

# run_jobs LOGDIR  <  lines "name|command": at most one job per GPU at a time; CUDA_VISIBLE_DEVICES is set per job.
run_jobs() {
  local logdir=$1 i=0 fail=0 line name cmd; mkdir -p "$logdir"; local -a pids=()
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    name=${line%%|*}; cmd=${line#*|}
    if [ ${#pids[@]} -ge "$NGPU" ]; then wait "${pids[0]}" || fail=1; pids=("${pids[@]:1}"); fi
    local gpu=${GPU_LIST[$((i % NGPU))]}; i=$((i + 1))
    say "  $name -> gpu $gpu"
    ( CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$logdir/$name.log" 2>&1 || { echo "FAILED" >> "$logdir/$name.log"; exit 1; } ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || { say "some jobs FAILED, see $logdir/*.log (grep -l FAILED)"; return 1; }
}

# write_recon_cfg CFG SIGMA CKPT IMAGE_DIR OUT_ROOT NSEEDS SIDS...
write_recon_cfg() {
  local cfg=$1 sigma=$2 ck=$3 img=$4 out=$5 nseeds=$6; shift 6
  local sids; sids=$(printf '"%s",' "$@"); sids="[${sids%,}]"
  mkdir -p "$(dirname "$cfg")"
  cat > "$cfg" <<YAML
# DPS reconstructions of the toy experiments: identity forward model y = x + sigma n, scale 0.5, 1000 steps skip 25, eta 1
sampler:
  name: additive_dps
  params: {sigma: $sigma, scale: 0.5, skip: 25, eta: 1.0, trajectory_steps: 1000, beta_start: 0.0001, beta_end: 0.02}
model_backbone: ddad
ckpt: $ck
test_origin_override: ddad_dps_whole
image_dir: $img
samples: $sids
num_seeds: $nseeds
out_root: $out
save_float: true
bottom_suffix: "4"
YAML
}
need_prior() {  # dataset -> path of its checkpoint, or exit with instructions
  local ds=$1; [ "$ds" = quadp_p50 ] && ds=quad_L1
  local ck="$CKPT/$ds/$EPOCH"
  [ -f "$ck" ] || { echo "missing prior $ck: train it with CUDA_VISIBLE_DEVICES=<gpu> python toy/train_prior.py --dataset $ds (or bash toy/train_priors.sh)" >&2; exit 2; }
  echo "$ck"
}
