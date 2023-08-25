# # export CUDA_VISIBLE_DEVICES=0

# zero-shot
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/llama_7b/ \
    --model_name_or_path output/flan_v2_7B/subsamp_model \
    --tokenizer_name_or_path output/flan_v2_7B/subsamp_model \
    --adapter_path output/flan_v2_7B/checkpoint-2000 \
    --eval_batch_size 2 \
    --load_in_8bit \
    --use_chat_format

# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama_7b_pretrained/ \
#     --model_name_or_path output/flan_v2_7B/subsamp_model \
#     --tokenizer_name_or_path output/flan_v2_7B/subsamp_model \
#     --eval_batch_size 2 \
#     --load_in_8bit \
#     --use_chat_format

# # zero-shot with chatgpt
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-0shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # few-shot with chatgpt
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-5shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # zero-shot with gpt4
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-0shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20


# # few-shot with gpt4
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-5shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 2