#!/usr/bin/env bash
set -u
set -o pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 {Ramani2017|Lee2019|Tan2021A|Tan2021B|Wu2024} PHYSICAL_GPU_ID" >&2
    exit 2
fi

dataset="$1"
physical_gpu_id="$2"
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "$dataset" in
    Ramani2017|Lee2019|Tan2021A|Tan2021B|Wu2024)
        config_name="$dataset"
        ;;
    *)
        echo "unknown dataset: $dataset" >&2
        exit 2
        ;;
esac

config_path="$project_dir/configs/$config_name.json"

if [ ! -f "$config_path" ]; then
    echo "unknown dataset: $dataset" >&2
    exit 2
fi

cd "$project_dir"
experiment_dir=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["save_path"])' "$config_path")
mkdir -p "$experiment_dir"
cp "$config_path" "$experiment_dir/config.requested.json"
date --iso-8601=seconds > "$experiment_dir/started_at.txt"

export CUDA_VISIBLE_DEVICES="$physical_gpu_id"
export PYTHONUNBUFFERED=1
python -m hicformer.train_inference --config "$config_path" 2>&1 | tee "$experiment_dir/train.log"
experiment_exit_code=${PIPESTATUS[0]}

printf '%s\n' "$experiment_exit_code" > "$experiment_dir/exit_code.txt"
date --iso-8601=seconds > "$experiment_dir/finished_at.txt"
exit "$experiment_exit_code"
