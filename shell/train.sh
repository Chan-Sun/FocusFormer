#!/usr/bin/env bash
# Usage: bash shell/train.sh <embryo|cisd> <fold 1-5> <r50|hiera_s> [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."

dataset="${1:-embryo}"
split="${2:-1}"
backbone="${3:-r50}"
shift $(( $# < 3 ? $# : 3 ))

python train.py \
  --config-file "configs/focusformer_${backbone}.yaml" \
  --dataset "$dataset" --split "$split" \
  --data_root "${DATA_ROOT:-data}" \
  --output_dir "work_dirs/${dataset}/${backbone}/split_${split}" \
  --num-gpus "${NUM_GPUS:-1}" \
  "$@"
