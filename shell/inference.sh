#!/usr/bin/env bash
# Usage: bash shell/inference.sh <embryo|cisd> <fold 1-5> <r50|hiera_s> [extra args]
# Evaluates work_dirs/<dataset>/<backbone>/split_<fold>/model_final.pth on the test split.
set -euo pipefail
cd "$(dirname "$0")/.."

dataset="${1:-embryo}"
split="${2:-1}"
backbone="${3:-r50}"
shift $(( $# < 3 ? $# : 3 ))
run_dir="work_dirs/${dataset}/${backbone}/split_${split}"

python test.py \
  --weights "${WEIGHTS:-${run_dir}/model_final.pth}" \
  --split "$split" \
  --data_root "${DATA_ROOT:-data}" \
  --output_dir "${run_dir}/eval" \
  --num-gpus "${NUM_GPUS:-1}" \
  "$@"
