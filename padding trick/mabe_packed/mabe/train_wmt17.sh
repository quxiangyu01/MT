#! /usr/bin/env bash
set -e

SEED=${1:-0}
STEPS=${2:-100000}
INTERVAL=${3:-5000}
DATA_DIR=../../envs/wmt17zh2en_fairseq

nohup python -u mabe.py \
    --src_lang zh \
    --trg_lang en \
    --src_bpe_fname $DATA_DIR/src_data_bin \
    --trg_bpe_fname $DATA_DIR/tgt_data_bin \
    --src_train_fname $DATA_DIR/train.clean.zh.id \
    --trg_train_fname $DATA_DIR/train.clean.en.id \
    --dim 512 \
    --num_layers 6 \
    --num_heads 8 \
    --dropout 0.3 \
    --max_epochs 100 \
    --max_steps $((STEPS+10)) \
    --accum_interval 20 \
    --train_batch_size 5000 \
    --save_dir log \
    --saving_interval $INTERVAL \
    --seed $SEED \
>train.out 2>&1 &
#2>&1 | tee train.out  &

echo launched training process "$!"
