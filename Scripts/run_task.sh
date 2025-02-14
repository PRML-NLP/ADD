python Self_Plan/Traj_Syn/run_task.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --llm_name llama-2-7b-chat-hf \
    --max_context_len 4096 \
    --task Hotpotqa \
    --task_path Self_Instruct/Meta_sample/Meta_Hotpotqa.json \
    --save_path Self_Plan/Traj_Syn/output/hotpotqa_train_data.jsonl\
    --use_openai False \
    --openai_key EMPTY
