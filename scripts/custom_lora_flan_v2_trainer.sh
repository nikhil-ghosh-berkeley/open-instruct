MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
SUBSAMP_RATIO=1.0
MAX_SEQ_LENGTH=512
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
#  --with_deepspeed ds_configs/stage3_offloading.conf \
python open_instruct/prepare_subsampled_model.py \
    --pretrained_model_name_or_path huggyllama/llama-7b \
    --tokenizer_name huggyllama/llama-7b \
    --use_fast_tokenizer False \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --double_quant \
    --quant_type nf4 \
    --bits 4 \
    --gradient_checkpointing \
    --train_file data/processed/flan_v2/flan_v2_data.jsonl \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --subsamp_ratio ${SUBSAMP_RATIO} \
    --output_dir output/flan_v2_${MODEL_SIZE}/ \
    --bf16 \
    --tf32 True \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --eval_dataset_size 1000 \
    --num_training_eval_samples 1000 \
    --remove_unused_columns True \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --report_to wandb

