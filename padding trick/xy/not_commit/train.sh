#! /usr/bin/env bash
set -e

SEED=${1:-0}
STEPS=${2:-100000}
INTERVAL=${3:-5000}
DATA_DIR=../../envs/wmt14ende_standford_mosesbpe37k

nohup python -u mt_trainer.py \
    --src_lang en \
    --src_bpe_fname $DATA_DIR/bpe.share.37000 \
    --trg_lang de \
    --trg_bpe_fname $DATA_DIR/bpe.share.37000 \
    --src_train_fname $DATA_DIR/train.en.id \
    --trg_train_fname $DATA_DIR/train.de.id \
    --src_valid_fname $DATA_DIR/val.en.id \
    --trg_valid_fname $DATA_DIR/val.de.id \
    --src_test_fname $DATA_DIR/test.en.id \
    --trg_test_fname $DATA_DIR/test.en.id \
    --dim 512 \
    --num_layers 6 \
    --num_heads 8 \
    --dropout 0.1 \
    --accum_interval 5 \
    --saving_interval $INTERVAL \
    --max_epochs 10 \
    --max_steps $((STEPS+10)) \
    --train_batch_size 5000 \
    --valid_batch_size 5000 \
    --test_batch_size 1024 \
    --save_dir logp \
    --seed $SEED \
>train.out 2>&1 &
#2>&1 | tee train.out  &

echo launched training process "$!"
