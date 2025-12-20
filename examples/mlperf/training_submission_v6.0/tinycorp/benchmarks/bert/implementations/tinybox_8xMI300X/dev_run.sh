#!/bin/bash

export PYTHONPATH="." AMD=1 AMD_LLVM=0
export MODEL="bert"
export DEFAULT_FLOAT="HALF" GPUS=1 BS=128 EVAL_BS=128
export DEFAULT_FLOAT="HALF" GPUS=1 BS=128 EVAL_BS=128
# export DEFAULT_FLOAT="HALF" GPUS=1 BS=192 EVAL_BS=192
# export DEFAULT_FLOAT="HALF" GPUS=8 BS=1024 EVAL_BS=1024
# export DEFAULT_FLOAT="HALF" GPUS=6 BS=768 EVAL_BS=768
# export DEFAULT_FLOAT="HALF" GPUS=1 BS=192 EVAL_BS=192
# export DEFAULT_FLOAT="HALF" GPUS=2 BS=192 EVAL_BS=192
# export DEFAULT_FLOAT="FLOAT" BS=128 EVAL_BS=128

# similar to https://github.com/mlcommons/training_results_v3.1/blob/d06288b2bd675a9d88e0e6181f5bb5626b71ec19/Quanta_Cloud_Technology/results/D54U-3U/bert/result_1.txt#L54
export OPT_BASE_LEARNING_RATE=0.0011 OPT_LAMB_BETA_1=0.60466 OPT_LAMB_BETA_2=0.85437 DECAY=0.1
# export OPT_LAMB_BETA_1=0.9 OPT_LAMB_BETA_2=0.999
# export OPT_BASE_LEARNING_RATE=0.0022 OPT_LAMB_BETA_1=0.60466 OPT_LAMB_BETA_2=0.85437 DECAY=0.1
export TRAIN_STEPS=3900

export IGNORE_OOB=1
export REWRITE_STACK_LIMIT=5000000

export BEAM=3 BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
# export BEAM=3 BEAM_UPCAST_MAX=2048 BEAM_LOCAL_MAX=1024 # 6756.08ms 6767.67 ms
# export BEAM_LOCAL_MAX=1024 BEAM_UPCAST_MAX=2048 BEAM=4
# export BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1 FREE_INTERMEDIATE=0
export BASEDIR="/raid/datasets/wiki"
# export IGNORE_BEAM_CACHE=1

export WANDB=1 PARALLEL=0
export FP8=1
export CUSTOM_CLAMP=0
export CUSTOM_AMAX=0
# export LOSS_SCALER=512
# export DEBUG=4

export TRAIN_STEPS=3900 
# export WANDB_NAME="clamp=1/amax=1 (6.18) 8/1024 mi350"
export WANDB_NAME="clamp=0/amax=0 (6.) 8/1024 mi350"
# export FP8_EXTRA=1 
# RUNMLPERF=1 python3 examples/mlperf/model_train.py
export HCQDEV_WAIT_TIMEOUT_MS=600000
# export TRAIN_STEPS=3900
# WANDB_NAME="clamp=1/amax=1 (6.) 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
export FP8_EXTRA=0 
# WANDB_NAME="clamp=1/amax=1 (6.18) 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# FP8=0 WANDB_NAME="FP8=0 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# RUNMLPERF=1 python3 examples/mlperf/model_train.py 2>&1| tee a1.txt
# WANDB_NAME="FP8=0 1/192 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# WANDB_NAME="qk 0/0 axis (%2 6-16) 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# sleep 120
# WANDB_NAME="clamp=0/amax=0 axis (%2 6-16) 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# WANDB_NAME="block-128 (%2 4-18) 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# b128 (block-128)
# export HCQ_VISIBLE_DEVICES="2,3,4,5,6,7" 
# FP8=1 TC128=0 WANDB_NAME="relu b128 (%2 4-18) 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# FP8=0 WANDB_NAME="FP8=0 conS 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py # FAIL
# FP8=3 WANDB_NAME="conS FP8=3 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=1 WANDB_NAME="conS FP8=1 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# FP8=2 WANDB_NAME="conS FP8=2 8/1024 mi350" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# export HCQ_VISIBLE_DEVICES="6,7" 
export WANDB=1
# TC128=1 GPUS=1 BS=128 EVAL_BS=128 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=1 WANDB_NAME="1/128 FP8=1 TC128=1 output" RUNMLPERF=1 python3 examples/mlperf/model_train.py
TC128=1 GPUS=8 BS=1024 EVAL_BS=1024 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=1 WANDB_NAME="1-22 TC128=1 allCON+output+im 8/1024 FP8=1" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# TC128=0 GPUS=8 BS=1024 EVAL_BS=1024 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=1 WANDB_NAME="1-22 TC128=0 allCON+output+im 8/1024 FP8=1" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# TC128=0 GPUS=8 BS=1024 EVAL_BS=1024 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=0 WANDB_NAME="FP8=0 allCON 1220" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# unset HCQ_VISIBLE_DEVICES
# export HCQ_VISIBLE_DEVICES="4,5,6,7" 
# TC128=1 GPUS=4 BS=512 EVAL_BS=512 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=0 WANDB_NAME="4/512 FP8=0 TC128=1 output" RUNMLPERF=1 python3 examples/mlperf/model_train.py
# TC128=1 GPUS=2 BS=128 EVAL_BS=128 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=1 WANDB_NAME="1/1 TC128=1 output 1/128 " RUNMLPERF=1 python3 examples/mlperf/model_train.py

# TC128=0 GPUS=1 BS=128 EVAL_BS=128 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=1 WANDB_NAME="output (only cont relu) 1/128 1/1 " RUNMLPERF=1 python3 examples/mlperf/model_train.py
# TC128=1 GPUS=1 BS=192 EVAL_BS=192 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=2 WANDB_NAME="TC128=1 output+intermediate 1/192 1/1 " RUNMLPERF=1 python3 examples/mlperf/model_train.py
# TC128=1 GPUS=1 BS=128 EVAL_BS=128 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=4 WANDB_NAME="output+intermediate 1/128 1/1 " RUNMLPERF=1 python3 examples/mlperf/model_train.py # GOOD
# TC128=1 GPUS=1 BS=128 EVAL_BS=128 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=4 WANDB_NAME="FP8=4 1/128 1/1 " RUNMLPERF=1 python3 examples/mlperf/model_train.py
# TC128=1 GPUS=1 BS=128 EVAL_BS=128 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=2 WANDB_NAME="TC128=1 FP8=2 1/128 1/1 " RUNMLPERF=1 python3 examples/mlperf/model_train.py
# GPUS=1 BS=128 EVAL_BS=128 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=2 WANDB_NAME="FP8=2 1/128 1/1 " RUNMLPERF=1 python3 examples/mlperf/model_train.py
# GPUS=1 BS=192 EVAL_BS=192 TRAIN_STEPS=5000 CUSTOM_CLAMP=1 CUSTOM_AMAX=1 FP8=1 WANDB_NAME="1/192 1/1 FP8=1 " RUNMLPERF=1 python3 examples/mlperf/model_train.py

# GPUS=1 BS=192 EVAL_BS=192 TRAIN_STEPS=5000 CUSTOM_CLAMP=0 CUSTOM_AMAX=0 FP8=0 WANDB_NAME="1/192 FP8=0 " RUNMLPERF=1 python3 examples/mlperf/model_train.py


# TC128 = 1 # 3732.44 ms AMD * 6,  4.41 loss
#    10 3882.71 ms run,  145.41 ms python,   4.86 ms fetch data, 3732.44 ms AMD * 6,  4.41 loss, 0.001097 LR, global_norm:  2.18, 1462.30 GB used, 197692.71 GFLOPS                                      
# TC128 = 0 # 3654.16 ms AMD * 6,  4.32 loss
#    10 3799.85 ms run,  141.31 ms python,   4.38 ms fetch data, 3654.16 ms AMD * 6,  4.32 loss, 0.001097 LR, global_norm:  1.24, 1462.30 GB used, 202003.55 GFLOPS                                      
# TC128=0 , relu
# 10 2287.55 ms run,  151.79 ms python,   4.50 ms fetch data, 2131.26 ms AMD * 6,  4.31 loss, 0.001097 LR, global_norm:  1.56, 1539.61 GB used, 380546.37 GFLOPS

# TC128=1 relu
# 10 2332.67 ms run,  140.62 ms python,   4.55 ms fetch data, 2187.50 ms AMD * 6,  4.07 loss, 0.001097 LR, global_norm:  1.58, 1539.61 GB used, 373185.20 GFLOPS                                      
#    11 2332.73 ms run,  141.54 ms python,   4.34 ms fetch data, 2186.86 ms AMD * 6,  4.26 loss, 0.001097 LR, global_norm:  1.89, 1539.61 GB used, 373174.88 GFLOPS

# 8/128 TC128=1 relu
# 10 2404.77 ms run,  210.21 ms python,   5.51 ms fetch data, 2189.05 ms AMD * 8,  4.20 loss, 0.001097 LR, global_norm:  1.35, 2054.72 GB used, 482663.60 GFLOPS 


# (baseline) relu FP8=0
# 60 2094.76 ms run,  206.31 ms python,   6.11 ms fetch data, 1882.33 ms AMD * 8,  4.11 loss, 0.001083 LR, global_norm:  0.82, 1462.94 GB used, 547824.29 GFLOPS 

# relu TC128=0
# 10 2347.61 ms run,  211.94 ms python,   5.86 ms fetch data, 2129.81 ms AMD * 8,  4.36 loss, 0.001097 LR, global_norm:  1.46, 2054.72 GB used, 494415.73 GFLOPS 