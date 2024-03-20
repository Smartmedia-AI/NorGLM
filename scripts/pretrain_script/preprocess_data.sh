#!/bin/bash

DATA_PATH=<Specify path to the training corpus>
OUTPUT_PATH=<Specify the output path>


python /Megatron-LM/tools/preprocess_data.py --input $DATA_PATH --output-prefix $OUTPUT_PATH --vocab /tokenizer/vocab.json --dataset-impl mmap --tokenizer-type GPT2BPETokenizer --merge-file /tokenizer/merges.txt --append-eod --workers 4 --chunk-size 512
