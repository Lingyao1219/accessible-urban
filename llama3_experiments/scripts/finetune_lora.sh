#!/bin/bash
GPUS_PER_NODE=2
CUDA_VISIBLE_DEVICES=0,1
LORA_RANK=32
LORA_ALPHA=$((LORA_RANK/2))
BATCH_SIZE=64
ACCU=$((BATCH_SIZE/$GPUS_PER_NODE))
EPOCH=10
LR=3e-5

SAVE_PATH=accessibility_llama3-8b-lora${LORA_RANK}_alpha${LORA_ALPHA}_ep${EPOCH}_bs${BATCH_SIZE}_lr${LR}
MODEL_PATH=<Path_to_your_downloaded_Meta-Llama-3-8B-Instruct-HF> # change this to the path where you downloaded the model


torchrun --nnodes 1 --nproc_per_node $GPUS_PER_NODE --node_rank 0 --master_addr localhost --master_port 29504 \
    train.py \
    --lora_enable True --lora_r ${LORA_RANK} --lora_alpha ${LORA_ALPHA} --lora_dropout 0.05 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version llama3 \
    --data_path data/train_llava_format.json \
    --bf16 True \
    --group_by_modality_length True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${ACCU} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1e5 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.08 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5120 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH}
    