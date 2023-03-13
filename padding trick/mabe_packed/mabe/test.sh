#! /usr/bin/env bash
set -e

INTERVAL=${1:-1500}
NUM_ITERATION=${2:-4}
GPU=${3:-0}
ENV=${4:-"wmt14"}

export CUDA_VISIBLE_DEVICES=${GPU}


for ((ITERATION=1; ITERATION<=$NUM_ITERATION; ITERATION++)); do
	STEP=$((ITERATION*INTERVAL))
	echo ${STEP}

  while [ ! -f log/checkpoints/*.${STEP}.ckpt ]; do
      # echo "didn't find checkpoint file"
      sleep 60
  done

  if [ -f ${STEP}.ckpt ]; then
      # echo "checkpoint file already copied"
      continue
  else
      sleep 60
      cp log/checkpoints/*.${STEP}.ckpt ${STEP}.ckpt

      for BS in {1,}; do
          echo "launched testing with beam size=${BS} for model ${STEP}.ckpt"
          #nohup python -u search.py ${BS} ./${STEP}.ckpt >${STEP}_bs${BS}.out
          nohup python -u probability_model_analysis.py $ENV ./${STEP}.ckpt logP vbs ${BS} >${STEP}_bs${BS}.out
          #mv vanilla_beam_search_bs${BS}.de ${STEP}_vbs${BS}.de
      done
      #break
  fi
done

python plot_learning_curve.py ${INTERVAL} ${NUM_ITERATION}
