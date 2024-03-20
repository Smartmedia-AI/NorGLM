#!/bin/bash

# Runs the "23B" parameter model in a distributed way on multi-nodes

GPUS_PER_NODE=<Specify the number of GPUS pre node>
MASTER_ADDR=<Specify the address of the host running the master process>
MASTER_PORT=<Specify the port on the master host>
NNODES=<Specify the number of nodes>
NODE_RANK=<Specify the rank of this node>
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))



DATA_PATH=<Specify path and file prefix>_text_document
CHECKPOINT_PATH=<Specify path>

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"



python -m torch.distributed.launch $DISTRIBUTED_ARGS /Megatron-DeepSpeed-main/pretrain_gpt.py \
       --num-layers 49 \
       --hidden-size 6144 \
       --num-attention-heads 64 \
       --micro-batch-size 4 \
       --global-batch-size 112 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --lr-decay-iters 10000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /tokenizer/vocab.json \
       --merge-file /tokenizer/merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --pipeline-model-parallel-size 7 \
       --tensor-model-parallel-size 4 \
       --deepspeed_config /ds_zero_stage_3.config \
       --checkpoint-activations \
       --checkpoint-num-layers 1 \
       --partition-activations \
       --synchronize-each-layer \
       --distributed-backend nccl \
       --lr 0.000097 \
       --lr-decay-style cosine \
       --min-lr 0.0000097 \
       --weight-decay 0.01 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --loss-scale 0 \
       --loss-scale-window 1000 \
       --hysteresis 2 \
       --min-loss-scale 1
