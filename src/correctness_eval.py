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
    correctness_eval_template,
)
from dataset_utils import JsonListWriter, JsonListReader
from backend_utils import start_backend, kill_all_my_gpu_processes, sgl
from collections import defaultdict

random.seed(int(time.time()*10))

PORT = 14284 + random.randint(0, 2000)
dataset_name = 'MMLU/MMLUPRO/Others shipped with MMLU'
data_volume = 32079 
data_source = 'test'
difficulties = 'all'
types = 'all'

per_field = defaultdict(lambda: [])

def correctness_eval(in_json_path: str, old_out_json_path: str, new_out_json_path: str, model_repoid_or_path: str, temperature: float):
    print(f"\n\nGetting responses for {model_repoid_or_path}, path={in_json_path} -> {new_out_json_path}\n", flush=True)
    global PORT
    
    with open(in_json_path, 'r') as f:
        input_data = json.load(f)
    
    with open(old_out_json_path, 'r') as f:
        existing_content = json.load(f)
    
    if isinstance(existing_content, list) and len(existing_content) > 5 and isinstance(existing_content[-1], dict) and 'reports' in existing_content[-1]:
        print(f'Found existing content in {old_out_json_path}, length={len(existing_content)}. Will build on top of it.')
        if len(existing_content) >= len(input_data):
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
        s += sgl.gen("NA", max_tokens=5, choices=['Matched Option/Answer', 'Mismatched Option/Answer'], temperature=1e-8)

    total_failures = 0
    
    with JsonListWriter(new_out_json_path) as writer:

        # Divide the data into batches
        num_batches = 200
        
        for i in tqdm.tqdm(range(num_batches)):
            
            batch_l, batch_r = i * len(existing_content) // num_batches, (i + 1) * len(existing_content) // num_batches
            
            dialogues = []
            for j in range(batch_l, batch_r):
                for k in range(2):
                    
                    simplified_dict = {
                        'question': existing_content[j]['question'],
                        'solution': existing_content[j]['solution'],
                        'student_answer': existing_content[j]['reports'][k]
                    }
                
                    prompt = deepcopy(correctness_eval_template)
                    prompt['content'] = prompt['content'].format(
                        content=json.dumps(simplified_dict)
                    )
                    
                    dialogue = [
                        question_system_prompt_identityless,
                        prompt,
                    ]
                    
                    dialogues.append(dialogue)
                    # if (j, k) == (0, 0):
                    #     print(f"Example dialogue: {dialogue}")
            
            output = get_response.run_batch([
                {"conversation": dialogue} for dialogue in dialogues
            ], progress_bar=True)
            
            count = 0
            while True:
                old_count = count
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
                
                if count == old_count:
                    print(f"Failed to get response for {count} cases")
                    break
            
            reports = []
            for ps in output:
                if ps.get_meta_info("NA") is None:
                    # print(f"Failed to get response for {ps}")
                    total_failures += 1
                    reports.append(1)
                    continue
                
                text = ps['NA']
                
                assert text in ['Matched Option/Answer', 'Mismatched Option/Answer']
                
                if text == 'Matched Option/Answer':
                    reports.append(1)
                else:
                    reports.append(0)
                
            assert len(reports) == 2 * (batch_r - batch_l)
            
            for j in range(batch_l, batch_r):
                new_case = deepcopy(existing_content[j])
                new_case['correctness'] = np.mean(reports[:2])
                per_field[new_case['type']].append(new_case['correctness'])
                reports = reports[2:]
                writer.append(new_case)
            
            assert len(reports) == 0
    
    print(f"Killing backend for {model_repoid_or_path}")
    backend.kill()
    kill_all_my_gpu_processes()
    print(f"Killed backend for {model_repoid_or_path}")
    
    print(f"Total failures: {total_failures}")
                        


if __name__ == '__main__':
    freeze_support()
    timestamp = '20240912-effectiveness-both'
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"results/{timestamp}", exist_ok=True)
    
    if os.environ.get('BASE_MODEL', ''):
        timestamp += f'-{os.environ["BASE_MODEL"].strip().lower()}'
    
    base_model_path = 'unsloth/gemma-2-27b-it'
    base_model_id = '27B'
    
    print("Starting experiment\n\n", flush=True)
    
    fileid = os.environ.get('FILEID')
        
    p = multiprocessing.Process(
        target=correctness_eval,
        args=(
            'cases-both.json', 
            f"results/{timestamp}/responses-{fileid}.json",
            f"results/{timestamp}/responses-{fileid}-eval.json",
            base_model_path,
            0.1
        ),
    )
    p.start()
    p.join()
    
    with open(f"results/{timestamp}/per_field_{fileid}.json", 'w') as f:
        json.dump(dict(per_field), f)