#! /usr/bin/env bash
set -e

SEED=${1:-0}
STEPS=${2:-300000}
INTERVAL=${3:-500}
DATA_DIR=../../envs/wmt17zh2en_fairseq
src=zh
tgt=en

nohup python -u mt_trainer.py \
    --src_lang $src \
    --src_bpe_fname $DATA_DIR/src_data_bin \
    --trg_lang $tgt \
    --trg_bpe_fname $DATA_DIR/tgt_data_bin \
    --src_train_fname $DATA_DIR/train.clean.$src.id \
    --trg_train_fname $DATA_DIR/train.clean.$tgt.id \
    --src_valid_fname $DATA_DIR/valid.clean.$src.id \
    --trg_valid_fname $DATA_DIR/valid.clean.$tgt.id \
    --src_test_fname $DATA_DIR/test.$src.id \
    --trg_test_fname $DATA_DIR/test.$tgt.id \
    --dim 512 \
    --num_layers 6 \
    --num_heads 8 \
    --dropout 0.3 \
    --accum_interval 20 \
    --saving_interval $INTERVAL \
    --max_epochs 50 \
    --max_steps $((STEPS+10)) \
    --train_batch_size 5000 \
    --valid_batch_size 5000 \
    --test_batch_size 1024 \
    --save_dir log \
    --seed $SEED \
>train.out 2>&1 &
#2>&1 | tee train.out  &

echo launched training process "$!"
