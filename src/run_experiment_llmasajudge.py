import numpy as np
import gc
import json
import time
import os
import tqdm
import random
from typing import List, Tuple, Literal, Optional
import multiprocessing
from multiprocessing import freeze_support
from copy import deepcopy
from templates import (
    question_system_prompt_identityless, 
    question_system_prompt_identityless_deceptive,
    prediction_elicitation_template_prompted_identityless,
    prediction_elicitation_template_unprompted_identityless,
    prediction_system_prompt_identityless,
    llm_as_a_judge_template,
    llm_as_a_judge_system_prompt,
)
from dataset_utils import JsonListWriter, JsonListReader
import random

random.seed(int(time.time()*10))
from backend_utils import start_backend, kill_all_my_gpu_processes, sgl

PORT = 14284 + random.randint(0, 2000)
dataset_name = 'MMLU/MMLUPRO/Others shipped with MMLU'
data_volume = 32079 
data_source = 'test'
difficulties = 'all'
types = 'all'

def get_all_llmasajudge(in_json_paths: List[Tuple[str, str]], out_json_path: str, model_repoid_or_path: str, predictor_name: str):
    print(f"\n\nGetting LLM-as-a-judge scores using {model_repoid_or_path}\n", flush=True)
    
    identities = [identity for _, identity in in_json_paths]
    data_files = []
    for data_path, _ in in_json_paths:
        with open(data_path, 'r') as f:
            data_files.append(json.load(f))
    
    min_len = min(len(data) for data in data_files)
    
    try:
        with open(out_json_path, 'r') as f:
            existing_content = json.load(f)
        
        if isinstance(existing_content, list) and len(existing_content) > 5 and isinstance(existing_content[-1], dict) and 'logprobs' in existing_content[-1]:
            print(f'Found existing content in {out_json_path}, length={len(existing_content)}. Will build on top of it.')
            if len(existing_content) >= min_len:
                print(f"Already finished {model_repoid_or_path}")
                return
    
    except:
        existing_content = []
    
    # Start backend without printing to stdout
    backend = start_backend(model_repoid_or_path, 'responses', port=PORT)

    @sgl.function
    def get_evalscores(s, conversation: List):
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
        s += 'Score (1-10): '
        s += sgl.gen("NA", max_tokens=5, return_logprob=False, temperature=0)
    
    def find_integer_in_string(s: str, min_val: int, max_val: int) -> Optional[int]:
        if s is None:
            return None

        nums = list(range(min_val, max_val + 1))[::-1]
        result = []
        for num in nums:
            if str(num) in s:
                result.append(num)
        
        if len(result) == 0:
            return None

        purged = s.replace(str(result[0]), '')
        for num in result[1:]:
            if str(num) in purged:
                return None
        
        return result[0]
    
    with JsonListWriter(out_json_path) as writer:
        
        # assert all(len(data) == len(data_files[0]) for data in data_files)
        
        # Divide the data into batches
        num_batches = 100
        max_batches_this_run = 100
        total_missing = 0
        
        for i in tqdm.tqdm(range(min(num_batches, max_batches_this_run)), position=0):
            
            batch_l, batch_r = i * min_len // num_batches, (i + 1) * min_len // num_batches
            
            if len(existing_content) >= batch_r:
                print(f"Skipping batch {i} because it is already in the output file")
                for j in range(batch_l, batch_r):
                    writer.append(existing_content[j], flush=(j == batch_r - 1))
                
                continue
            
            dialogues = []
            batch_start_time = time.time()
            
            for j in tqdm.tqdm(range(batch_l, batch_r), position=1):
                for informant in range(len(data_files)):
                    evaluation_prompt = deepcopy(llm_as_a_judge_template)
                    evaluation_prompt['content'] = evaluation_prompt['content'].format(
                        question=data_files[0][j]['question'],
                        response=data_files[informant][j]['reports'][0],
                    )
                    
                    dialogues.append([
                        llm_as_a_judge_system_prompt,
                        evaluation_prompt,
                    ])
            
            output = get_evalscores.run_batch([
                {"conversation": dialogue} for dialogue in dialogues
            ], progress_bar=True)
            
            for _ in range(10):
                count = 0
                for k in tqdm.tqdm(range(len(output)), position=2):
                    if output[k].get_meta_info("NA") is None:
                        output[k] = get_evalscores.run_batch([
                            {"conversation": dialogues[k]}
                        ])[0]
                        count += 1
                
                print(f"Re-run {count} cases")
                if count == 0:
                    break
                
            batch_end_time = time.time()
            print('\n\n\n', flush=True)
            print(f"Finished batch {i}/{max_batches_this_run}={i / max_batches_this_run * 100:.2f}%, time={batch_end_time - batch_start_time:.2f}s, estimated time remaining={(batch_end_time - batch_start_time) * (max_batches_this_run - i - 1):.2f}s\n\n\n", flush=True)
            
            eval_reports = [
                (ps['NA'] if ps.get_meta_info("NA") is not None else None)
                for ps in output
            ]
            assert len(eval_reports) == len(dialogues)
            
            for j in range(batch_l, batch_r):
                informant_performances = []
                skip = False
                
                for informant in range(len(data_files)):
                    
                    score = find_integer_in_string(eval_reports[0], 1, 10)
                    eval_report_summary = str(eval_reports[0]).replace('\n', '\\n')[:20]
                    
                    if eval_reports[0] is None:
                        print(f"Missing eval_report for {j} {informant}")
                        total_missing += 1
                        skip = True
                    
                    elif 'sorry, but' in data_files[informant][j]['reports'][0].lower() or 'sorry, but' in data_files[informant][j]['reports'][1].lower() or \
                       'false but' in data_files[informant][j]['reports'][0].lower() or 'false but' in data_files[informant][j]['reports'][1].lower():
                        print(f"Skipping {j} {informant} because it contains 'sorry, but' or 'false but': score {eval_reports[0]}") #, {data_files[informant][j]['reports'][0]}")
                        total_missing += 1
                        skip = True
                    
                    elif score is None:
                        print(f"Missing score for {j} {informant}: {eval_report_summary}")
                        total_missing += 1
                        skip = True
                    
                    if j == batch_l and not skip:
                        print(f"Score for {j} {informant}: {score} | {eval_report_summary}")
                    
                    eval_reports = eval_reports[1:]
                    if skip:
                        continue    
                    
                    informant_performances.append((
                        informant,
                        score,
                        data_files[informant][j]['reports'],
                        random.random(),
                    ))
                
                if skip:
                    writer.append(data_files[0][j], flush=(j == batch_r - 1))
                    continue
                
                sorted_performance = list(sorted(informant_performances, key=lambda x: (x[1], x[3]), reverse=True))
                
                new_case = deepcopy(data_files[0][j])
                del new_case['reports']
                new_case['sorted_reports'] = sorted_performance
                writer.append(new_case, flush=(j == batch_r - 1))
    
    print(f"Finished {model_repoid_or_path}, missing {total_missing} elements")
    print(f"Killing backend for {model_repoid_or_path}")
    backend.kill()
    kill_all_my_gpu_processes()
    print(f"Killed backend for {model_repoid_or_path}")
                        


if __name__ == '__main__':
    freeze_support()
    timestamp = (
        '20240912-witness-both-dedup-nowild-n4-mixed' if os.environ.get('MIXED', '0') == '1' else
        '20240912-witness-both-n4' if os.environ.get('DUP', '0') == '1' else 
        '20240912-witness-both-dedup-nowild-n4'
    )
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    if os.environ.get('MISLEAD', '0') == '1':
        timestamp +=  '-mislead'
    
    if os.environ.get('BASE_MODEL', ''):
        timestamp += f'-{os.environ["BASE_MODEL"].strip().lower()}'
        
    os.makedirs(f"results/{timestamp}", exist_ok=True)
    
    if os.environ.get('FORMAL', '') == '1':
        timestamp = '20240912-formallies'
        print('Using formal lies')
    
    
    base_model_path = '/nas/models/Meta-Llama-3.1-8B-Instruct'
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
    elif os.environ.get('BASE_MODEL', '').lower() == 'gemma2-9b':
        base_model_path = 'unsloth/gemma-2-9b-it'
        base_model_id = '9B'
    elif os.environ.get('BASE_MODEL', '').lower() == 'mistral-7b':
        base_model_path = '/nas/models/Mistral-7B-Instruct-v0.3'
        base_model_id = '7B'
    elif os.environ.get('BASE_MODEL', '').lower() != '':
        raise ValueError(f"Unknown base model: {os.environ['BASE_MODEL']}")
    
    models = [(base_model_path, f'{base_model_id}-{i}') for i in range(1, 17)]
    
    predictor_models = [
        ('/nas/models/Mistral-7B-Instruct-v0.3', 'mistral-7B'),
        ('Qwen/Qwen2.5-7B-Instruct', 'qwen2.5-7B'),
        ('Qwen/Qwen2.5-3B-Instruct', 'qwen2.5-3B'),
        ('Qwen/Qwen2.5-1.5B-Instruct', 'qwen2.5-1.5B'),
        ('Qwen/Qwen2.5-0.5B-Instruct', 'qwen2.5-0.5B'),
        ('HuggingFaceTB/SmolLM-360M-Instruct', 'smol-360M'),
        ('HuggingFaceTB/SmolLM-135M-Instruct', 'smol-135M'),
    ]
    
    predictor_model_id = int(os.environ.get('PREDICTOR_MODEL_ID', '0'))
    predictor_models = [predictor_models[predictor_model_id]]
    
    if os.environ.get('NUM_MODELS', '') != '':
        models = models[:int(os.environ['NUM_MODELS'])]
        print(f"Using {len(models)} models")
    
    assert os.environ.get('NUM_DECEPTIVE_MODELS') is not None
    num_deceptive_models = int(os.environ.get('NUM_DECEPTIVE_MODELS'))
    print('NUM_DECEPTIVE_MODELS:', num_deceptive_models)
    
    for i in range(int(os.environ.get('DECEPTIVE_START_IND', '0')), num_deceptive_models):
        if os.environ.get('RECONSTRUCT', '0') == '1':
            models[i] = (models[i][0], models[i][1] + '-reconstructeddeceptive')
        else:
            models[i] = (models[i][0], models[i][1] + '-hyperdeceptive')
    
    predictor_models = [
        (model_repoid_or_path, model_name + f'-{num_deceptive_models}-{len(models)}' + ('-reconstructed' if os.environ.get('RECONSTRUCT', '0') == '1' else '') + (f'-start{os.environ.get("DECEPTIVE_START_IND", 0)}' if os.environ.get('DECEPTIVE_START_IND', 0) != 0 else ''))
        for model_repoid_or_path, model_name in predictor_models
    ]
    
    if os.environ.get('MIXED', '') == '1':
        models = [
            ('unsloth/gemma-2-9b-it', '9B-1-hyperdeceptive'),
            ('unsloth/gemma-2-9b-it', '9B-2'),
            ('/nas/models/Mistral-7B-Instruct-v0.3', '7B-3-hyperdeceptive'),
            ('/nas/models/Mistral-7B-Instruct-v0.3', '7B-4'),
            ('/nas/models/Meta-Llama-3.1-8B-Instruct', '8B-5-hyperdeceptive'),
            ('/nas/models/Meta-Llama-3.1-8B-Instruct', '8B-6'),
        ]
    
    print('predictor_models:', predictor_models)
    print('models:', models)
    
    print("Starting experiment\n\n", flush=True)
    
    for model_repoid_or_path, model_name in models:
        assert os.path.exists(f"results/{timestamp}/responses-{model_name}.json")
    
    print("\n\n\nGetting logprobs\n\n", flush=True)
    
    for model_repoid_or_path, model_name in predictor_models:
        PORT += 1
        p = multiprocessing.Process(
            target=get_all_llmasajudge,
            args=[
                [
                    (f"results/{timestamp}/responses-{other_model_name}.json", other_model_name)
                    for _, other_model_name in models
                ],
                f"results/{timestamp}/llmasajudge-{model_name}.json",
                model_repoid_or_path,
                model_name,
            ],
        )
        print(f"Starting {model_name}")
        p.start()
        print(f"Started {model_name}")
        p.join()
        print(f"Finished {model_name}")
        time.sleep(20)