import numpy as np
import gc
import json
import time
import os
import tqdm
import random
from typing import List, Tuple, Literal
import multiprocessing
from multiprocessing import freeze_support
from copy import deepcopy
from templates import (
    question_system_prompt_identityless, 
    question_system_prompt_identityless_deceptive,
    prediction_elicitation_template_prompted_identityless,
    prediction_elicitation_template_unprompted_identityless,
    prediction_system_prompt_identityless,
    answer_translation_template,
    answer_translation_prefix
)
from dataset_utils import JsonListWriter, JsonListReader
from backend_utils import start_backend, kill_all_my_gpu_processes, sgl

random.seed(int(time.time()*10))

PORT = 14284 + random.randint(0, 2000)
dataset_name = 'MMLU/MMLUPRO/Others shipped with MMLU'
data_volume = 32079 
data_source = 'test'
difficulties = 'all'
types = 'all'

def pretrain_to_instruct(in_json_path: str, old_out_json_path: str, new_out_json_path: str, model_repoid_or_path: str, temperature: float):
    print(f"\n\nGetting responses for {model_repoid_or_path}, path={in_json_path} -> {new_out_json_path}\n", flush=True)
    global PORT
    
    if 'hyperdeceptive' not in new_out_json_path:
        return
    
    with open(in_json_path, 'r') as f:
        data = json.load(f)
    
    with open(old_out_json_path, 'r') as f:
        existing_content = json.load(f)
    
    if isinstance(existing_content, list) and len(existing_content) > 5 and isinstance(existing_content[-1], dict) and 'reports' in existing_content[-1]:
        print(f'Found existing content in {old_out_json_path}, length={len(existing_content)}. Will build on top of it.')
        if len(existing_content) >= len(data):
            print(f"Already finished {model_repoid_or_path}")
            
    else:
        raise ValueError(f"Existing content in {old_out_json_path} is not completed.")
    
    # Start backend without printing to stdout
    backend = start_backend(model_repoid_or_path, 'responses', port=PORT)
    
    @sgl.function
    def get_response(s, conversation: List):
        nonlocal temperature, new_out_json_path
        
        prefix = ""
        for turn in conversation:
            if turn['role'] == 'assistant':
                s += sgl.assistant(turn['content'])
            elif turn['role'] == 'user':
                s += sgl.user(turn['content'])
            elif turn['role'] == 'system':
                s += sgl.system(turn['content'])
            elif turn['role'] == 'prefix':
                prefix = turn['content']
            else:
                raise ValueError(f"Unknown role: {turn['role']}")
        
        s += sgl.assistant_begin()
        s += prefix
        s += sgl.gen("NA", max_tokens=512, return_logprob=False, temperature=temperature)

    with JsonListWriter(new_out_json_path) as writer:

        # Divide the data into batches
        num_batches = 200
        
        for i in tqdm.tqdm(range(num_batches)):
            
            batch_l, batch_r = i * len(data) // num_batches, (i + 1) * len(data) // num_batches
            
            dialogues = []
            for j in range(batch_l, batch_r):
                for k in range(2):
                    
                    def json_repr(s: str) -> str:
                        s = repr(s)
                        s = '"' + s[1:-1].replace('"', '\\"') + '"'
                        return s
                
                    prompt = deepcopy(answer_translation_template)
                    prompt['content'] = prompt['content'].replace(
                        '{binary_choice_question}',
                        json_repr(existing_content[j]['question'])
                    ).replace(
                        '{multiple_choice_question}',
                        json_repr(data[j]['question'])
                    ).replace(
                        '{student_answer}',
                        json_repr(existing_content[j]['reports'][k])
                    )
                    
                    cur_answer_translation_prefix = answer_translation_prefix.replace(
                        '{binary_choice_question}',
                        json_repr(existing_content[j]['question'])
                    ).replace(
                        '{multiple_choice_question}',
                        json_repr(data[j]['question'])
                    ).replace(
                        '{student_answer}',
                        json_repr(existing_content[j]['reports'][k])
                    )
                    
                    dialogue = [
                        question_system_prompt_identityless,
                        prompt,
                        {
                            "role": "prefix",
                            "content": cur_answer_translation_prefix
                        }
                    ]
                    
                    dialogues.append(dialogue)
                    if (j, k) == (0, 0):
                        print(f"Example dialogue: {dialogue}")
                        print(f"Example prefix: {cur_answer_translation_prefix}")
            
            output = get_response.run_batch([
                {"conversation": dialogue} for dialogue in dialogues
            ], progress_bar=True)
            
            while True:
                count = 0
                for k in tqdm.tqdm(range(len(output)), position=2):
                    if output[k].get_meta_info("NA") is None:
                        output[k] = get_response.run_batch([
                            {"conversation": dialogues[k]}
                        ])[0]
                        count += 1
                
                print(f"Re-run {count} cases")
                if count == 0:
                    break
            
            reports = []
            for ps in output:
                text = ps['NA']
                
                if '```' in text:
                    text = text.split('```')[0]
                
                if '\"' in text and text.count('\"') == 1:
                    text = text.split('\"')[0]
                elif '\",' in text and text.count('\",') == 1:
                    text = text.split('\",')[0]
                elif '\",\n' in text and text.count('\",\n') == 1:
                    text = text.split('\",\n')[0]
                elif '\"\n' in text and text.count('\"\n') == 1:
                    text = text.split('\"\n')[0]
                elif '\n' in text and text.count('\n') == 1:
                    text = text.split('\n')[0]
                
                reports.append(text.strip())
                
            assert len(reports) == 2 * (batch_r - batch_l)
            
            sorted_lengths = sorted([len(report) for report in reports])
            print(f'Report lengths: {sorted_lengths[:-10][::10], sorted_lengths[-10:]}')
            
            for j in range(batch_l, batch_r):
                new_case = deepcopy(existing_content[j])
                new_case['reports'] = reports[:2]
                reports = reports[2:]
                writer.append(new_case)
            
            assert len(reports) == 0
    
    print(f"Killing backend for {model_repoid_or_path}")
    backend.kill()
    kill_all_my_gpu_processes()
    print(f"Killed backend for {model_repoid_or_path}")
                        


if __name__ == '__main__':
    freeze_support()
    timestamp = '20241026-train-deceptive'
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"results/{timestamp}", exist_ok=True)
    
    if os.environ.get('BASE_MODEL', ''):
        timestamp += f'-{os.environ["BASE_MODEL"].strip().lower()}'
    
    if os.environ.get('FORMAL', '') == '1':
        timestamp = '20240912-formallies'
        print('Using formal lies')
    
    os.makedirs(f"results/{timestamp}", exist_ok=True)
    
    base_model_path = '/models/Meta-Llama-3.1-8B-Instruct'
    base_model_id = '8B'
    
    if os.environ.get('BASE_MODEL', '').lower() == 'gemma2-2b':
        base_model_path = 'unsloth/gemma-2-2b-it'
        base_model_id = '2B'
    elif os.environ.get('BASE_MODEL', '').lower() == 'phi3.5-mini':
        base_model_path = 'microsoft/Phi-3.5-mini-instruct'
        base_model_id = '4B'
    elif os.environ.get('BASE_MODEL', '').lower() == 'gemma2-27b':
        base_model_path = 'unsloth/gemma-2-27b-it'
        base_model_id = '27B'
    elif os.environ.get('BASE_MODEL', '').lower() != '':
        raise ValueError(f"Unknown base model: {os.environ['BASE_MODEL']}")
    
    models = [(base_model_path, f'{base_model_id}-{i}') for i in range(1, 17)]
    
    print("Models:", models)
    print("Starting experiment\n\n", flush=True)
    
    fileid = int(os.environ.get('FILEID', '1'))
        
    p = multiprocessing.Process(
        target=pretrain_to_instruct,
        args=(
            'cases-both.json', 
            f"results/20241026-train-deceptive/responses-Mislead-7B-{fileid}.json",
            f"results/20241026-train-deceptive-llama2-7b/responses-7B-{fileid}-hyperdeceptive.json", 
            models[0][0],
            0.3
        ),
    )
    p.start()
    p.join()