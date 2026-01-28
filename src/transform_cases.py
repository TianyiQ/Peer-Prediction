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
    transform_cases_template,
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

def transform_cases(in_json_path: str, out_json_path: str, model_repoid_or_path: str, temperature: float):
    print(f"\n\nGetting responses for {model_repoid_or_path}, path={in_json_path} -> {out_json_path}\n", flush=True)
    global PORT
    
    with open(in_json_path, 'r') as f:
        data = json.load(f)
    
    # Start backend without printing to stdout
    backend = start_backend(model_repoid_or_path, 'responses', port=PORT)
    
    @sgl.function
    def get_response(s, conversation: List):
        nonlocal temperature, out_json_path
        
        for turn in conversation:
            if turn['role'] == 'assistant':
                s += sgl.assistant(turn['content'])
            elif turn['role'] == 'user':
                s += sgl.user(turn['content'])
            elif turn['role'] == 'system':
                s += sgl.system(turn['content'])
            else:
                raise ValueError(f"Unknown role: {turn['role']}")
        
        s += sgl.assistant_begin()
        s += 'Here is the reformatted question, strictly following the format "Story:\n[background_info]\n\nQuestion: [question]\nAnswer A: [option1]\nAnswer B: [option2]".\n\n```\n'
        s += sgl.gen("NA", max_tokens=16384, return_logprob=False, temperature=temperature)

    with JsonListWriter(out_json_path) as writer:

        # Divide the data into batches
        num_batches = 100
        
        for i in tqdm.tqdm(range(num_batches)):
            
            batch_l, batch_r = i * len(data) // num_batches, (i + 1) * len(data) // num_batches
            
            dialogues = []
            for j in range(batch_l, batch_r):
                
                prompt = deepcopy(transform_cases_template)
                prompt['content'] = prompt['content'].format(content=data[j]['question'])
                
                dialogue = [
                    question_system_prompt_identityless,
                    prompt,
                ]
                
                dialogues.append(dialogue)
            
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
                    text = text.split('```')[0].strip().strip('"')
                else:
                    text = text.strip().strip('"')
                
                reports.append(text)
                
            assert len(reports) == (batch_r - batch_l)
            
            sorted_lengths = sorted([len(report) for report in reports])
            print(f'Question lengths: {sorted_lengths[:-10][::10], sorted_lengths[-10:]}')
            
            for j in range(batch_l, batch_r):
                new_case = deepcopy(data[j])
                new_case['question'] = reports[j - batch_l]
                writer.append(new_case)
    
    print(f"Killing backend for {model_repoid_or_path}")
    backend.kill()
    kill_all_my_gpu_processes()
    print(f"Killed backend for {model_repoid_or_path}")
                     
def enforce_formatting():
    with open('cases-both-transformed-final.json', 'r') as f:
        data = json.load(f)
    
    for i in range(len(data)):
        question = data[i]['question']
        
        forbidden_strings = [
            'Background:',
            'Background info:',
            'Background information:',
            'Background information',
            'Background info',
            'Background Info:',
            'Background Information:',
            'Background Information',
        ]
        
        for string in forbidden_strings:
            for string2 in [string, string.replace(' ', '-'), string.replace(' ', '_')]:
                question = question.replace(string2, 'Story:')
        
        if question.strip()[:6] != 'Story:':
            
            if 'Story:' in question:
                question = 'Story:'.join(question.split('Story:')[1:])
            
            question = question.replace('Story:', '')
            assert 'Story:' not in question
            question = 'Story:\n' + question
            
            for _ in range(10):
                question = question.replace('\n\n\n', '\n\n')
                question = question.replace('Story:\n\n', 'Story:\n')
            
            # print(question)
            # input()
        
        for _ in range(10):
            question = question.replace('\n\n\n', '\n\n')
            question = question.replace('Story:\n\n', 'Story:\n')
        
        data[i]['question'] = question
    
    with open('cases-both-transformed-final2.json', 'w') as f:
        json.dump(data, f, indent=2)   

def truncate_lengths():
    with open('cases-both-transformed.json', 'r') as f:
        data = json.load(f)
    
    sample_ids_sorted = sorted(range(len(data)), key=lambda i: len(data[i]['question']), reverse=True)
    
    for i in range(len(sample_ids_sorted)):
        print(data[sample_ids_sorted[i]]['question'])
        truncate_string = input('Truncate to: ').strip()
        
        if truncate_string == '[exit]':
            break
        
        if truncate_string == '[skip]':
            continue
        
        assert truncate_string in data[sample_ids_sorted[i]]['question']
        data[sample_ids_sorted[i]]['question'] = data[sample_ids_sorted[i]]['question'].split(truncate_string)[0].strip()
        print(data[sample_ids_sorted[i]]['question'], '\n\n\n')
        input()
    
    with open('cases-both-transformed-final.json', 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    freeze_support()
    
    base_model_path = '/nas/models/Meta-Llama-3.1-8B-Instruct'
    base_model_id = '8B'
    
    enforce_formatting()
    # truncate_lengths()
    
    # p = multiprocessing.Process(
    #     target=transform_cases,
    #     args=('cases-both.json', 'cases-both-transformed.json', base_model_path, 0.25),
    # )
    # p.start()
    # p.join()
    # print('Done.')