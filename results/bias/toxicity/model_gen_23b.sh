#!/bin/bash
source /cluster/home/penl/workspace/instructionGPT/bin/activate
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.3.1


CUDA_VISIBLE_DEVICES=0,1,2,3 python generateFromModel.py


