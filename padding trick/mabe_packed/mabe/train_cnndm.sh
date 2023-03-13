#! /usr/bin/env bash
set -x

SEED=${1:-0}
STEPS=${2:-100000}
INTERVAL=${3:-5000}
DATA_DIR=../../envs/cnn_summarization

nohup python -u mabe.py \
    --src_lang source \
    --trg_lang target \
    --src_bpe_fname $DATA_DIR/src_data_bin \
    --trg_bpe_fname $DATA_DIR/tgt_data_bin \
    --src_train_fname $DATA_DIR/train.source.id \
    --trg_train_fname $DATA_DIR/train.target.id \
    --dim 768 \
    --num_layers 6 \
    --num_heads 8 \
    --dropout 0.1 \
    --max_epochs 100 \
    --max_steps $((STEPS+10)) \
    --accum_interval 10 \
    --train_batch_size 8000 \
    --save_dir log \
    --saving_interval $INTERVAL \
    --seed $SEED \
>train.out 2>&1 &
#2>&1 | tee train.out  &

echo launched training process "$!"
