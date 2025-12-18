#!/bin/bash

export PYTHONPATH="." AMD=1 AMD_LLVM=0
export MODEL="bert"
# export DEFAULT_FLOAT="HALF" GPUS=8 BS=1024 EVAL_BS=1024
export DEFAULT_FLOAT="HALF" GPUS=1 BS=192 EVAL_BS=192

# similar to https://github.com/mlcommons/training_results_v3.1/blob/d06288b2bd675a9d88e0e6181f5bb5626b71ec19/Quanta_Cloud_Technology/results/D54U-3U/bert/result_1.txt#L54
export OPT_BASE_LEARNING_RATE=0.0011 OPT_LAMB_BETA_1=0.60466 OPT_LAMB_BETA_2=0.85437 DECAY=0.1
# export TRAIN_STEPS=3900
export BENCHMARK=10 BERT_LAYERS=24

export IGNORE_OOB=1
export REWRITE_STACK_LIMIT=500000

export BEAM=3 BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
# export BEAM=4 BEAM_UPCAST_MAX=2048 BEAM_LOCAL_MAX=1024 # 6756.08ms 6767.67 ms
# export BEAM_LOCAL_MAX=1024 BEAM_UPCAST_MAX=2048 BEAM=4
# export BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1 FREE_INTERMEDIATE=0
export BASEDIR="/raid/datasets/wiki"

export WANDB=0 PARALLEL=0
export FP8=1
# export LOSS_SCALER=512
# export DEBUG=4
RUNMLPERF=1 python3 examples/mlperf/model_train.py
# RUNMLPERF=1 python3 examples/mlperf/model_train.py 2>&1| tee a1.txt