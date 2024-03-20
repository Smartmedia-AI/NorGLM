#!/bin/bash

#Runs the "3B" parameter model in a single node

DATA_PATH=<Specify path to the pre-processed training dataset>
OUTPUT_PATH=<Specify the model save path>

deepspeed /TencentPretrain/pretrain.py \
      --deepspeed \
      --deepspeed_config /TencentPretrain/deepspeed_zero3_config.json \
      --dataset_path $DATA_PATH \
      --spm_model_path /TencentPretrain/llama/tokenizer/nor_llama.model  \
      --vocab_path /TencentPretrain/llama/tokenizer/nor_llama.vocab \
      --deep_init  \
      --config_path /TencentPretrain/3b_config.json \
      --output_model_path $OUTPUT_PATH \
      --world_size 4 \
      --learning_rate 0.000056 \
      --gpu_ranks 0 1 2 3 \
      --total_steps 1000000 \
      --save_checkpoint_steps 10000 \
      --report_steps 1000