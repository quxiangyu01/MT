#! /usr/bin/env bash
set -x

SEED=${1:-0}
STEPS=${2:-6000}
INTERVAL=${3:-1500}
DATA_DIR=../../envs/wmt14ende_standford_mosesbpe37k

nohup python -u mabe.py \
    --src_lang en \
    --trg_lang de \
    --src_bpe_fname $DATA_DIR/bpe.share.37000 \
    --trg_bpe_fname $DATA_DIR/bpe.share.37000 \
    --src_train_fname $DATA_DIR/train.en.id \
    --trg_train_fname $DATA_DIR/train.de.id \
    --dim 512 \
    --num_layers 6 \
    --num_heads 8 \
    --dropout 0.1 \
    --max_epochs 100 \
    --max_steps $((STEPS+10)) \
    --accum_interval 5 \
    --train_batch_size 5000 \
    --save_dir log \
    --saving_interval $INTERVAL \
    --seed $SEED \
>train.out 2>&1 &
#2>&1 | tee train.out  &

echo launched training process "$!"
