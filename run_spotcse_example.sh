#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. See README.md for installation." >&2
    exit 1
fi
if [[ ! -f .venv/bin/activate ]]; then
    echo "Run 'uv sync --locked' first." >&2
    exit 1
fi
source .venv/bin/activate

TRAIN_FILE="${SPOTCSE_TRAIN_FILE:-data/wiki1m_for_simcse.txt}"
OUTPUT_DIR="${SPOTCSE_OUTPUT_DIR:-result/spotcse}"
MODEL_NAME="${SPOTCSE_MODEL:-bert-base-uncased}"
export CUDA_VISIBLE_DEVICES="${SPOTCSE_GPU:-0}"

if [[ ! -s "$TRAIN_FILE" ]]; then
    echo "Training data not found: $TRAIN_FILE. Run '(cd data && bash download_wiki.sh)'." >&2
    exit 1
fi
if [[ ! -f SentEval/data/downstream/STS/STSBenchmark/sts-dev.csv ]]; then
    echo "SentEval data not found. Run '(cd SentEval/data/downstream && bash download_dataset.sh)'." >&2
    exit 1
fi

env -u LD_LIBRARY_PATH uv run --locked python train.py \
    --model_name_or_path "$MODEL_NAME" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR/tensorboard" \
    --logging_steps 125 \
    --per_device_train_batch_size 64 \
    --num_train_epochs 1 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --save_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --temp 0.05 \
    --sinkhorn_temp 0.03 \
    --do_train \
    --do_eval \
    "$@"
