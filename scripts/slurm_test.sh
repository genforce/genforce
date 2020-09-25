#!/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
CHECKPOINT=$5
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_NODE=${CPUS_PER_NODE:-8}

SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:6}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ./test.py ${CONFIG} \
        --work_dir=${WORK_DIR} \
        --checkpoint ${CHECKPOINT} \
        --launcher="slurm" \
        ${PY_ARGS}
