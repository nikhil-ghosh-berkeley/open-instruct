MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# --deepspeed ds_configs/stage3_offloading.conf \
python open_instruct/custom_finetune_trainer.py \
    --model_name_or_path output/flan_v2_7B/subsamp_model \
    --tokenizer_name output/flan_v2_7B/subsamp_model \
    --use_fast_tokenizer False \
    --train_file data/processed/flan_v2/flan_v2_data.jsonl \
    --max_seq_length 512 \
    --preprocessing_num_workers 16 \
    --do_train \
    --do_eval \
    --use_lora \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --eval_dataset_size 1000 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 2 \
    --num_train_epochs 3 \
    --output_dir output/flan_v2_${MODEL_SIZE}_master/ \
    --bf16 \
    --tf32 True \
    --report_to wandb \
