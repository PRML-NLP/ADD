import argparse
from pre_prompt import (
    DATA_GEN_SYSTEM_PROMPT,
    HOTPOTQA_TASK_NAME,
    HOTPOTQA_TASK_DESCRIPTION,
    HOTPOTQA_DATA_GEN_HUMAN_PROMPT,
    FILTERING_PROMPT,
    DATA_GEN_SYSTEM_PROMPT_WITH_SEED,
)
from llms import MetaAgent
import json
import random
import re
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(verbose=True)


def convert_json_to_jsonl(input_file):
    """
    Convert a JSON file containing a list of objects to a JSONL file.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSONL file.
    """
    if input_file.endswith('.json'):
        # .json -> .jsonl
        output_file = input_file[:-5] + '.jsonl'

    try:
        # Load the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure the JSON is a list
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of objects.")
        
        # Write to JSONL format
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f_out.write(json_line + '\n')
        
        print(f"Successfully converted {input_file} to {output_file}.")
    
    except Exception as e:
        print(f"Error: {e}")

def get_data_hotpotqa(source_data):
    data=json.load(open(source_data))
    data=[{"Question":d['Question'],"Answer":d['Answer']} for d in data]
    return data


def get_random_data(data, num_samples=1):
    random_data = random.sample(data, num_samples)

    return random_data


def parse_ouput_hotpotqa(output):
    pattern = r"Question: (.+)\nAnswer: (.+)"
    matches = re.findall(pattern, output)
    # print(matches)
    new_qa_pairs=[]
    for match in matches:
        question = match[0]
        answer = match[1]
        new_qa_pairs.append({
            'Question': question,
            'Answer': answer,
        })
    return new_qa_pairs

def save_to_json(data,path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as file:
            ori_data = json.load(file)
    else:
        ori_data = []
    with open(path, 'w') as file:
        data = data+ori_data
        json.dump(data, file,indent=4)
        
def main(args):
    data_system_prompt = DATA_GEN_SYSTEM_PROMPT

    # 어떤 task인지 정하고 해당 system prompt 가져오기
    if args.dataset_name == "hotpotqa":
        dataset_system_prompt = data_system_prompt.format(task_name = HOTPOTQA_TASK_NAME, task_description = HOTPOTQA_TASK_DESCRIPTION)

    meta_agent = MetaAgent(
        use_openai=args.use_openai,
        model_name=args.model_name,
        url=args.openai_base,
        system_prompt= dataset_system_prompt 
    )

    qa_pairs=[]
    if args.dataset_name=="hotpotqa":
        qa_pairs=get_data_hotpotqa(args.source_data)


    #else dataset_name
    answer_set=set()
    unique_qa=[]
    ori_qa=get_random_data(qa_pairs,num_samples=2)
    unique_qa = ori_qa
    for u in unique_qa:
        if args.dataset_name == "hotpotqa":
            answer_set.add(u["Answer"])

    iteration = 0
    while(len(answer_set)<args.generate_all_num):
        print(f"{args.generate_per_round_num}x{iteration} iteration, have generated num {len(answer_set)}, all {args.generate_all_num} need to be generated all")
        all_qa=""
        # max 5 examples
        sample_pairs=get_random_data(ori_qa,num_samples=2)+get_random_data(unique_qa,num_samples=min(3,len(unique_qa)))
        random.shuffle(sample_pairs)

        for qa in sample_pairs:
            all_qa+=str(qa)[1:-1]+"\n"
        human_prompt_args = {"QA_pairs":all_qa,"Gen_num":args.generate_per_round_num}
        human_prompt_template = HOTPOTQA_DATA_GEN_HUMAN_PROMPT if args.dataset_name == "hotpotqa" else SCIENCEQA_DATA_GEN_HUMAN_PROMPT
        output = meta_agent.generate(
            human_prompt_template=human_prompt_template,
            human_prompt_args=human_prompt_args,
            # temprature=random.uniform(0.1, 0.5),
            # top_k=args.top_k,
            # top_p=args.top_p,
            # max_tokens=args.max_tokens,
            update_prompt=False
        )
        iteration += 1
        
        data_list = []
        if(args.dataset_name == "hotpotqa"):
            for new_pair in parse_ouput_hotpotqa(output):
                # print(new_pair,'\n')
                
                template = f"Question: {new_pair['Question']}\n Answer: {new_pair['Answer']}"
                messages = [{"role":"system", "content": FILTERING_PROMPT}, {"role":"user", "content": template}]
                chat_completion = ChatOpenAI(model="gpt-4o")
                ret = chat_completion.invoke(messages).content
                print(ret)
                correctness = True if ret == "True" or ret == "true" else False
                
                if new_pair["Answer"] not in answer_set and len(new_pair["Answer"])<20 and correctness:
                    unique_qa.append(new_pair)
                    answer_set.add(new_pair["Answer"])
                    data_list.append(new_pair)
            save_to_json(data_list,args.target_data)


    # convert json to jsonl
    convert_json_to_jsonl(args.target_data)
            

            
        
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--source_data", type=str, default="/workspace/Self_instruct/Meta_sample/Meta_Hotpotqa.json")
    parser.add_argument("--generate_all_num", type=int, default=10)
    parser.add_argument("--generate_per_round_num", type=int, default=10)
    parser.add_argument("--target_data",type=str,default="/workspace/generated_qa.json")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--openai_key", type=str, default="EMPTY")
    parser.add_argument("--openai_base", type=str, default="http://localhost:23333/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--use_openai", type=bool, default=False)
    args = parser.parse_args()
    
    main(args)