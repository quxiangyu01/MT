#! /usr/bin/env bash
set -x

DEVICES=${1:-0}
MODEL_PATH=${2:-"./mle_100k_seed0.ckpt"}
MODEL_OUTPUT=${3:-"logP"}
LOG_FOLDER=${4:-"logP"}
ENV=${5-"wmt14"}

mkdir -p $LOG_FOLDER

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT empty > $LOG_FOLDER/empty.out 2>&1

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT ref > $LOG_FOLDER/ref.out 2>&1

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT sample  1.0 > $LOG_FOLDER/sample_1.0.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  4 > $LOG_FOLDER/vbs_4.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  8 > $LOG_FOLDER/vbs_8.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  1 > $LOG_FOLDER/vbs_1.out 2>&1

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT sample  0.75 > $LOG_FOLDER/sample_0.75.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT sample  0.5 > $LOG_FOLDER/sample_0.5.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT sample  0.25 > $LOG_FOLDER/sample_0.25.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT sample  0.01 > $LOG_FOLDER/sample_0.01.out 2>&1

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  2 > $LOG_FOLDER/vbs_2.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  16 > $LOG_FOLDER/vbs_16.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  32 > $LOG_FOLDER/vbs_32.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  64 > $LOG_FOLDER/vbs_64.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  128 > $LOG_FOLDER/vbs_128.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  256 > $LOG_FOLDER/vbs_256.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  512 256 > $LOG_FOLDER/vbs_512.out 2>&1
CUDA_VISIBLE_DEVICES=${DEVICES} nohup python -u probability_model_analysis.py $ENV $MODEL_PATH $MODEL_OUTPUT vbs  1024 256 > $LOG_FOLDER/vbs_1024.out 2>&1

python plot_mle_analysis.py $LOG_FOLDER
