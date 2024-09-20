#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python corruptor_main.py --name concat_all \
     --resume ./logs/train_concat_all/checkpoints \
     --base configs/imagenet_pix2pix.yaml --train --gpus 0,1,2,3,4,5,6,7