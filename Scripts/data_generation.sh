python Self_Instruct/data_generation.py \
    --source_data Self_Instruct/Meta_sample/Meta_Hotpotqa.json \
    --target_data Self_Instruct/generated_metaqa.json \
    --dataset_name hotpotqa  \
    --generate_all_num 5000 \
    --generate_per_round_num 10 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --use_api false \
