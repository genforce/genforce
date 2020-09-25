#!/bin/bash

GPUS=$1
CONFIG=$2
WORK_DIR=$3
CHECKPOINT=$4
PORT=${PORT:-29500}

python -m torch.distributed.launch \
       --nproc_per_node=${GPUS} \
       --master_port=${PORT} \
       ./test.py ${CONFIG} \
           --work_dir ${WORK_DIR} \
           --checkpoint ${CHECKPOINT} \
           --launcher="pytorch" \
           ${@:5}
