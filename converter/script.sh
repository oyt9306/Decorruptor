#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python convert_LDM_main.py --checkpoint_path ../logs/train_concat_all/checkpoints/last.ckpt \
     --original_config_file ../configs/imagenet_pix2pix.yaml \
     --config_files ../configs/imagenet_pix2pix.yaml \
     --num_in_channels 8 \
     --half --to_safetensors \
     --dump_path ./save_decorruptor_dpm --extract_ema
