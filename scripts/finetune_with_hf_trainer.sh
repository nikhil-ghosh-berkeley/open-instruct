MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed open_instruct/finetune_trainer.py \
    --deepspeed ds_configs/stage3_offloading.conf \
    --model_name_or_path huggyllama/llama-7b \
    --tokenizer_name huggyllama/llama-7b \
    --use_fast_tokenizer False \
    --train_file data/processed/flan_v2/flan_v2_data.jsonl \
    --max_seq_length 512 \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --num_train_epochs 3 \
    --output_dir output/flan_v2_${MODEL_SIZE}_master/ \
    --bf16 \
    --tf32 True \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --overwrite_output_dir \
    --report_to wandb
