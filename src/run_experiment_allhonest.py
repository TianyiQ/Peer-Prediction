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
    prediction_elicitation_template_prompted_identityless_once,
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

def get_all_responses(in_json_path: str, out_json_path: str, model_repoid_or_path: str, temperature: float):
    print(f"\n\nGetting responses for {model_repoid_or_path}, path={in_json_path} -> {out_json_path}\n", flush=True)
    global PORT
    
    with open(in_json_path, 'r') as f:
        data = json.load(f)
    
    try:
        with open(out_json_path, 'r') as f:
            existing_content = json.load(f)
        
        if isinstance(existing_content, list) and len(existing_content) > 5 and isinstance(existing_content[-1], dict) and 'reports' in existing_content[-1]:
            print(f'Found existing content in {out_json_path}, length={len(existing_content)}. Will build on top of it.')
            if len(existing_content) >= len(data):
                print(f"Already finished {model_repoid_or_path}")
                return
        elif not isinstance(existing_content, list):
            print(f"Existing content is not a list, will start from scratch")
            existing_content = []
        elif len(existing_content) <= 5:
            print(f"Existing content is too short, will start from scratch")
            existing_content = []
        elif not isinstance(existing_content[-1], dict):
            print(f"Existing content is not a list of dictionaries, will start from scratch")
            existing_content = []
        elif 'reports' not in existing_content[-1]:
            print(f"Existing content does not have 'reports' field in the last element, will start from scratch")
            existing_content = []
    except:
        existing_content = []
    
    # Start backend without printing to stdout
    backend = start_backend(model_repoid_or_path, 'responses', port=PORT)
    
    @sgl.function
    def get_response(s, conversation: List):
        nonlocal temperature
        
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
        s += sgl.gen("NA", max_tokens=512, return_logprob=False, temperature=temperature)

    with JsonListWriter(out_json_path) as writer:

        # Divide the data into batches
        num_batches = 100
        
        for i in tqdm.tqdm(range(num_batches)):
            
            batch_l, batch_r = i * len(data) // num_batches, (i + 1) * len(data) // num_batches
            if len(existing_content) >= batch_r:
                print(f"Skipping batch {i} because it is already in the output file")
                for j in range(batch_l, batch_r):
                    writer.append(existing_content[j], flush=(j == batch_r - 1))
                
                continue
            
            batch = data[batch_l:batch_r]
            
            dialogues = []
            for case in batch:
                dialogue = [
                    question_system_prompt_identityless, 
                    {'role': 'user', 'content': case['question']},
                ]
                
                dialogues.append(dialogue)
                dialogues.append(deepcopy(dialogue))
            
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
            
            reports = [ps['NA'] for ps in output]
            assert len(reports) == 2 * len(batch)
            
            sorted_lengths = sorted([len(report) for report in reports])
            print(f'Report lengths: {sorted_lengths[:-10][::10], sorted_lengths[-10:]}')
            
            for i, case in enumerate(batch):
                new_case = deepcopy(case)
                new_case['reports'] = reports[2 * i: 2 * i + 2]
                writer.append(new_case)
    
    print(f"Killing backend for {model_repoid_or_path}")
    backend.kill()
    kill_all_my_gpu_processes()
    print(f"Killed backend for {model_repoid_or_path}")


def get_all_prediction_logprobs(in_json_paths: List[Tuple[str, str]], out_json_path: str, model_repoid_or_path: str, predictor_name: str):
    print(f"\n\nGetting prediction logprobs for {model_repoid_or_path}\n", flush=True)
    use_once = ('-once' in out_json_path)
    
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
    backend = start_backend(model_repoid_or_path, 'logprobs', port=PORT)

    @sgl.function
    def get_logprobs(s, conversation: List):
        for turn in conversation:
            if turn['role'] == 'assistant':
                s += sgl.assistant(turn['content'])
            elif turn['role'] == 'user':
                s += sgl.user(turn['content'])
            elif turn['role'] == 'system':
                s += sgl.system(turn['content'])
            else:
                raise ValueError(f"Unknown role: {turn['role']}")
                
        s += sgl.gen("NA", max_tokens=0, return_logprob=True, logprob_start_len=0)
    
    with JsonListWriter(out_json_path) as writer:
        
        # assert all(len(data) == len(data_files[0]) for data in data_files)
        
        # Divide the data into batches
        num_batches = 1000
        max_batches_this_run = 1000
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
                    for predictee in range(len(data_files)):
                        if predictee == predictor_name:
                            continue
                        
                        all_ref_questions = []
                        while len(all_ref_questions) < 3:
                            k = random.randint(0, min_len - 1)
                            while k in all_ref_questions:
                                k = random.randint(0, min_len - 1)
                            
                            if (('level' not in data_files[0][j] or 'level' not in data_files[0][k] or data_files[0][k]['level'] == data_files[0][j]['level']) and
                                data_files[0][k]['type'] == data_files[0][j]['type'] and
                                data_files[0][k]['question'] != data_files[0][j]['question']):
                                all_ref_questions.append(k)
                        
                        if use_once:
                            elicitation_prompt = deepcopy(prediction_elicitation_template_prompted_identityless_once)
                            elicitation_prompt['content'] = elicitation_prompt['content'].format(
                                question=data_files[0][j]['question'],
                                informant_answer=data_files[informant][j]['reports'][0],
                                reference_question0=data_files[0][all_ref_questions[0]]['question'],
                                reference_predictee_answer0=data_files[predictee][all_ref_questions[0]]['reports'][0],
                                reference_question1=data_files[0][all_ref_questions[1]]['question'],
                                reference_predictee_answer1=data_files[predictee][all_ref_questions[1]]['reports'][0],
                                reference_question2=data_files[0][all_ref_questions[2]]['question'],
                                reference_predictee_answer2=data_files[predictee][all_ref_questions[2]]['reports'][0],
                            )
                        
                        else:
                            elicitation_prompt = deepcopy(prediction_elicitation_template_prompted_identityless)
                            elicitation_prompt['content'] = elicitation_prompt['content'].format(
                                question=data_files[0][j]['question'],
                                informant_answer=data_files[informant][j]['reports'][0],
                                reference_question0=data_files[0][all_ref_questions[0]]['question'],
                                reference_informant_answer0=data_files[informant][all_ref_questions[0]]['reports'][0],
                                reference_predictee_answer0=data_files[predictee][all_ref_questions[0]]['reports'][0],
                                reference_question1=data_files[0][all_ref_questions[1]]['question'],
                                reference_informant_answer1=data_files[informant][all_ref_questions[1]]['reports'][0],
                                reference_predictee_answer1=data_files[predictee][all_ref_questions[1]]['reports'][0],
                                reference_question2=data_files[0][all_ref_questions[2]]['question'],
                                reference_informant_answer2=data_files[informant][all_ref_questions[2]]['reports'][0],
                                reference_predictee_answer2=data_files[predictee][all_ref_questions[2]]['reports'][0],
                            )
                        
                        dialogues.append([
                            prediction_system_prompt_identityless,
                            elicitation_prompt,
                        ])
                        
                        dialogues.append([
                            prediction_system_prompt_identityless,
                            elicitation_prompt,
                            {
                                'role': 'assistant', 
                                'content': data_files[predictee][j]['reports'][1],
                            }
                        ])
                        
                        
                        elicitation_prompt_unprompted = deepcopy(prediction_elicitation_template_unprompted_identityless)
                        elicitation_prompt_unprompted['content'] = elicitation_prompt_unprompted['content'].format(
                            question=data_files[0][j]['question'],
                            reference_question0=data_files[0][all_ref_questions[0]]['question'],
                            reference_predictee_answer0=data_files[predictee][all_ref_questions[0]]['reports'][0],
                            reference_question1=data_files[0][all_ref_questions[1]]['question'],
                            reference_predictee_answer1=data_files[predictee][all_ref_questions[1]]['reports'][0],
                            reference_question2=data_files[0][all_ref_questions[2]]['question'],
                            reference_predictee_answer2=data_files[predictee][all_ref_questions[2]]['reports'][0],
                        )
                        
                        dialogues.append([
                            prediction_system_prompt_identityless,
                            elicitation_prompt_unprompted,
                        ])
                        
                        dialogues.append([
                            prediction_system_prompt_identityless,
                            elicitation_prompt_unprompted,
                            {
                                'role': 'assistant', 
                                'content': data_files[predictee][j]['reports'][1],
                            }
                        ])
            
            output = get_logprobs.run_batch([
                {"conversation": dialogue} for dialogue in dialogues
            ], progress_bar=True)
            
            for _ in range(10):
                count = 0
                for k in tqdm.tqdm(range(len(output)), position=2):
                    if output[k].get_meta_info("NA") is None:
                        output[k] = get_logprobs.run_batch([
                            {"conversation": dialogues[k]}
                        ])[0]
                        count += 1
                
                print(f"Re-run {count} cases")
                if count == 0:
                    break
                
            batch_end_time = time.time()
            print('\n\n\n', flush=True)
            print(f"Finished batch {i}/{max_batches_this_run}={i / max_batches_this_run * 100:.2f}%, time={batch_end_time - batch_start_time:.2f}s, estimated time remaining={(batch_end_time - batch_start_time) * (max_batches_this_run - i - 1):.2f}s\n\n\n", flush=True)
            
            logprobs = []
            
            for ps in output:
                if ps.get_meta_info("NA") is None:
                    logprobs.append(None)
                else:
                    logprobs.append(
                        sum(x[0] for x in list(ps.get_meta_info("NA")['input_token_logprobs']) if x[0] is not None)
                    )
            
            for j in range(batch_l, batch_r):
                percase_results = {}
                
                for informant in range(len(data_files)):
                    for predictee in range(len(data_files)):
                        if predictee == predictor_name:
                            continue
                        
                        if any(logprobs[k] is None for k in range(4)):
                            total_missing += 1
                            print(f"{total_missing}-th: Missing logprobs for {j} {informant} {informant} {predictee}")
                            percase_results[f"{informant}_{predictor_name}_{predictee}"] = (None, [None] * 4)    
                        
                        else:
                            percase_results[f"{informant}_{predictor_name}_{predictee}"] = (
                                (logprobs[1] - logprobs[0]) - (logprobs[3] - logprobs[2]),
                                logprobs[:4],
                            )
                        
                        logprobs = logprobs[4:]
                
                new_case = deepcopy(data_files[0][j])
                del new_case['reports']
                new_case['logprobs'] = percase_results
                writer.append(new_case, flush=(j == batch_r - 1))
    
    print(f"Finished {model_repoid_or_path}, missing {total_missing} elements")
    print(f"Killing backend for {model_repoid_or_path}")
    backend.kill()
    kill_all_my_gpu_processes()
    print(f"Killed backend for {model_repoid_or_path}")
                        


if __name__ == '__main__':
    freeze_support()
    timestamp = '20240912-effectiveness-both'
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"results/{timestamp}", exist_ok=True)
    
    
    models = [
        ('/nas/models/Meta-Llama-3.1-8B-Instruct', '8B-1'),
        ('/nas/models/Meta-Llama-3.1-8B-Instruct', '8B-2'),
        ('/nas/models/Meta-Llama-3.1-8B-Instruct', '8B-3'),
    ]
    predictor_models = [
        ('/nas/models/Mistral-7B-Instruct-v0.3', 'mistral-7B'),
        # ('Qwen/Qwen2.5-7B-Instruct', 'qwen2.5-7B'),
        # ('Qwen/Qwen2.5-3B-Instruct', 'qwen2.5-3B'),
        # ('Qwen/Qwen2.5-1.5B-Instruct', 'qwen2.5-1.5B'),
        # ('Qwen/Qwen2.5-0.5B-Instruct', 'qwen2.5-0.5B'),
    ]
    
    if os.environ.get('ONCE', '0') == '1':
        predictor_models = [
            (model_repoid_or_path, model_name + '-once')
            for model_repoid_or_path, model_name in predictor_models
        ]
    
    print("Starting experiment\n\n", flush=True)
    
    for model_repoid_or_path, model_name in models:
        PORT += 1
        p = multiprocessing.Process(
            target=get_all_responses,
            args=('cases-both.json', f"results/{timestamp}/responses-{model_name}.json", model_repoid_or_path, 1.0),
        )
        p.start()
        p.join()
    
    print("\n\n\nGetting logprobs\n\n", flush=True)
    
    for model_repoid_or_path, model_name in predictor_models:
        PORT += 1
        p = multiprocessing.Process(
            target=get_all_prediction_logprobs,
            args=[
                [
                    (f"results/{timestamp}/responses-{other_model_name}.json", other_model_name)
                    for _, other_model_name in models
                ],
                f"results/{timestamp}/logprobs-{model_name}.json",
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
        
    with JsonListWriter(f"results/{timestamp}/sorted_reports.json") as writer:
        
        readers_responses = [
            JsonListReader(f"results/{timestamp}/responses-{model_name}.json").__enter__()
            for _, model_name in models
        ]
        
        readers_logprobs = [
            JsonListReader(f"results/{timestamp}/logprobs-{model_name}.json").__enter__()
            for _, model_name in predictor_models
        ]
             
        print("\n\nSorting reports\n\n", flush=True)
        
        for all_cases in tqdm.tqdm(list(zip(
            *readers_responses,
            *readers_logprobs,
        ))):
            cases = [
                (all_cases[i], models[i][1])
                for i in range(len(models))
            ]
            logprobs = [
                (all_cases[i + len(models)], predictor_models[i][1])
                for i in range(len(predictor_models))
            ]
            
            informant_performances = {}
            
            skip = False
            
            try:
                for informant_id, (informant_case, informant) in enumerate(cases):
                    informant_performances[informant] = [
                        0, # total logprobs
                        informant_case['reports'], # reports
                    ]
                    
                    for predictor_id, (data, predictor) in enumerate(logprobs):
                        for predictee_id, (_, predictee) in enumerate(models):
                            if predictor == predictee:
                                continue
                            
                            if f"{informant_id}_{predictor}_{predictee_id}" not in data['logprobs']:
                                print(f"Missing {informant_id}_{predictor}_{predictee_id} among {data['logprobs'].keys()}")
                                assert False
                            
                            informant_performances[informant][0] += data['logprobs'][f"{informant_id}_{predictor}_{predictee_id}"][0]
            except:
                print('Skipping case')
                skip = True
            
            if skip:
                writer.append(all_cases[0])
                continue
            
            # Sort the informant performances along with respective total logprobs
            sorted_performance = sorted(
                [(informant, info[0], info[1]) for informant, info in informant_performances.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            
            new_case = deepcopy(all_cases[0])
            del new_case['reports']
            new_case['sorted_reports'] = sorted_performance
            writer.append(new_case)