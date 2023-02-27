#! /usr/bin/env bash
set -e

GPU=${1:-1}
INTERVAL=${2:-1500}
NUM_ITERATION=${3:-4}

export CUDA_VISIBLE_DEVICES=${GPU}

#for ((ITERATION=0; ITERATION<=$NUM_ITERATION; ITERATION++)); do
for ITERATION in {1,${NUM_ITERATION}}; do
	STEP=$((ITERATION*INTERVAL))
	echo ${STEP}

  if [ ! -f log/checkpoints/*.${STEP}.ckpt ]; then
      # echo "didn't find checkpoint file"
      continue
  elif [ -f ${STEP}.ckpt ]; then
      # echo "checkpoint file already copied"
      continue
  else
      cp log/checkpoints/*.${STEP}.ckpt ${STEP}.ckpt
      export CUDA_VISIBLE_DEVICES=${GPU}

      for BS in {1,4,7,10,13,16}; do
          nohup python -u search.py ${BS} ./${STEP}.ckpt >${STEP}_bs${BS}.out &
          echo "launched testing with beam size=${BS} for model ${STEP}.ckpt"
      done
      break
  fi
done