#!/bin/bash

DATA_PATH=<Specify path to the training corpus>

python preprocess.py \
      --corpus_path $DATA_PATH \
      --spm_model_path llama/tokenizer/nor_llama.model \
      --dataset_path datasets/pretaining_dataset.pt \
      --processes_num 64 \
      --data_processor lm

