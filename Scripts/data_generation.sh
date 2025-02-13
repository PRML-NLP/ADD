
# Self-instruct 폴더로 이동
cd "$(dirname "$0")/../Self_instruct"

# Python 실행
PYTHONPATH=$(pwd) python -m data_generation \
    --source_data Meta_sample/Meta_Hotpotqa.json \
    --target_data generated_metaqa.json \
    --dataset_name hotpotqa \
    --generate_all_num 5000 \
    --generate_per_round_num 10 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --use_openai false
