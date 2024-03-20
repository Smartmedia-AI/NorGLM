#!/bin/bash

# Continue training the "3B" parameter model in a single node

GPUS_PER_NODE=<Specify the number of GPUS pre node>
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DATA_PATH=<Specify path and file prefix>_text_document
CHECKPOINT_PATH=<Specify path>

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"



deepspeed /Megatron-DeepSpeed-main/pretrain_gpt.py \
       --num-layers 32 \
       --hidden-size 2688 \
       --num-attention-heads 32 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 1390000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /tokenizer/vocab.json \
       --merge-file /tokenizer/merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --pipeline-model-parallel-size 1 \
       --tensor-model-parallel-size 1 \
       --deepspeed_config /ds_zero_stage_3.config \
       --checkpoint-activations \
       --checkpoint-num-layers 1 \
       --partition-activations \
       --synchronize-each-layer \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --loss-scale 0 \
       --loss-scale-window 1000 \
       --hysteresis 2 \
       --min-loss-scale 1
