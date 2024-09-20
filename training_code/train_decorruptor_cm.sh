#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
MODEL_DIR='path/to/your_DPM'
OUTPUT_DIR='imagenet_decorruptor_cm'

accelerate launch train_LCM_from_ckpt.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=256 \
    --learning_rate=1e-5 --loss_type="huber" \
    --adam_weight_decay=0.0 \
    --max_train_steps=500000 \
    --max_train_samples=40000000 \
    --dataloader_num_workers=8 \
    --validation_steps=500    \
    --num_ddim_timesteps=50 \
    --checkpointing_steps=4000 \
    --checkpoints_total_limit=10 \
    --train_batch_size=64 \
    --path_imagenet='path/to/your/imagenet_train_data' \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=24 