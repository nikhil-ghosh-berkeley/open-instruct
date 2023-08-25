export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path huggyllama/llama-7b \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --tokenizer_name huggyllama/llama-7b \
    --use_slow_tokenizer \
    --train_file data/processed/flan_v2/flan_v2_data.jsonl \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir output/alpaca_${MODEL_SIZE}_lora/ \
    --save_merged_lora_model \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1