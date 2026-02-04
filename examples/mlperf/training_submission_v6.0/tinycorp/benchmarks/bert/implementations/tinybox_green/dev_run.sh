#!/bin/bash

export PYTHONPATH="." NV=1
export MODEL="bert"
export DEFAULT_FLOAT="HALF" SUM_DTYPE="HALF" GPUS=6 BS=72 EVAL_BS=72
export DEFAULT_FLOAT="HALF" SUM_DTYPE="HALF" GPUS=4 BS=48 EVAL_BS=48

export CHECK_OOB=0
export REWRITE_STACK_LIMIT=500000

export BEAM=8 BEAM_UOPS_MAX=10000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1
export BASEDIR="/raid/datasets/wiki"
export FP8_TRAIN=0
# search
#IGNORE_BEAM_CACHE=0 BENCHMARK=10 BERT_LAYERS=2 RUNMLPERF=0 python3 examples/mlperf/model_train.py

export WANDB=1 PARALLEL=0

#RUNMLPERF=1 python3 examples/mlperf/model_train.py



export FP8_TRAIN=1
# search
IGNORE_BEAM_CACHE=0 BENCHMARK=10 BERT_LAYERS=2 RUNMLPERF=0 python3 examples/mlperf/model_train.py
IGNORE_BEAM_CACHE=0 BENCHMARK=10 RUNMLPERF=0 python3 examples/mlperf/model_train.py
RUNMLPERF=1 python3 examples/mlperf/model_train.py
