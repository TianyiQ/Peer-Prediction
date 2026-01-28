from collections import defaultdict
from get_raw_stats_counterfactual import get_metrc
import subprocess
import os
import json

participant_models = [
    ('8B', './results/20240912-witness-both-dedup-nowild-n4', None),
    ('2B', './results/20240912-witness-both-dedup-nowild-n4-gemma2-2b', 'gemma2-2b'),
    ('27B', './results/20240912-witness-both-dedup-nowild-n4-gemma2-27b', 'gemma2-27b'),
]

mix_num = None
if os.environ.get('MIXED', None) is not None:
    mix_num = int(os.environ.get('MIXED'))
    participant_models = [
        ('7B/8B/9B', f'./results/20240912-witness-both-dedup-nowild-n4-mixed', None),
    ]

models = [
    ('smol-135M', 6), 
    ('smol-360M', 5),
    ('qwen2.5-0.5B', 4), 
    ('qwen2.5-1.5B', 3), 
    ('qwen2.5-3B', 2),
    ('qwen2.5-7B', 1),
]

results = defaultdict(lambda: defaultdict(lambda: []))
output_path = f'./scaling-data{"-mixed" if os.environ.get("MIXED", None) else ""}.json'

if os.environ.get('MISLEAD', '0') == '1':
    output_path = output_path.replace('.json', '-mislead.json')
    participant_models = [
        (participant_models[0][0], participant_models[0][1] + '-mislead', None),
    ]

print(f'Output path: {output_path}')
print(f'Participant models: {participant_models}')

if os.environ.get('POWERSET', None) is not None:
    output_path = f'./scaling-data-powerset{"-mixed" if os.environ.get("MIXED", None) else ""}.json'
    
if os.environ.get('MULTIPLYSIZE', '0') != '0':
    output_path = output_path.replace('.json', f'-multi{os.environ.get("MULTIPLYSIZE", "0")}.json')
    
if os.environ.get('CELOSS', '0') == '1':
    output_path = output_path.replace('.json', '-celoss.json')

if os.environ.get('LINEAR', '0') == '1':
    output_path = output_path.replace('.json', '-linear.json')

if os.environ.get('LOGISTIC', '0') == '1':
    output_path = output_path.replace('.json', '-logistic.json')

if os.path.exists(output_path):
    with open(output_path) as f:
        results = json.load(f)
    
    print(f'Loaded results')

for participant_id, prefix, participant_base in participant_models:

    all_combinations = [[i] for i in range(len(models))]
    
    if os.environ.get('POWERSET', None) is not None:
        all_combinations = [
            [i for i in range(len(models)) if (j & (1 << i)) > 0]
            for j in range(1, 2**len(models))
        ]
        print(f'Running powerset for participant {participant_id}, {len(all_combinations)} combinations')
    
    model_iter = 0
    for combination in all_combinations:
        model_iter += 1
        
        if os.environ.get('POWERSET', None) is None:
            model, model_id = models[combination[0]]
    
            data_files = [
                (
                    'LLM-as-a-Judge',
                    [
                        (f'llmasajudge-{model}-1.json', None, None, None),
                        (f'llmasajudge-{model}-1-2.json', None, None, None),
                        (f'llmasajudge-{model}-1-1.json', None, None, None),
                    ],
                    2
                ),
                # (
                #     'LLM-as-a-Judge (6-shot)',
                #     [
                #         (f'llmasajudge-{model}-0-0.json', None, None, None),
                #     ],
                #     2
                # ),
            ] + [
                (
                    f'Peer Prediction (|P|={n})',
                    [
                        (f'sorted_reports-1-{n}.json', (None if participant_id == '7B/8B/9B' else f'sorted_reports-0-2.json'), f'logprobs-{model}-1-{n}.json', (None if participant_id == '7B/8B/9B' else f'logprobs-{model}-0-2.json')),
                        (f'part7/sorted_reports-1-{n}.json', f'part7/sorted_reports-0-2.json', f'part7/logprobs-{model}-part7-1-{n}.json', f'part7/logprobs-{model}-part7-0-2.json'),
                    ],
                    n
                )
                for n in ([2, 4, 8, 16] if participant_id != '7B/8B/9B' else [4])
            ]
        
        else:
            model_id = ','.join([str(models[i][1]) for i in combination])
            data_files = [
                (
                    f'Peer Prediction (|P|={n})',
                    [
                        (f'sorted_reports-1-{n}.json', f'sorted_reports-0-2.json', None, None)
                    ],
                    n
                )
                for n in [2, 4, 8]
            ]
        
        for name, files, n in data_files:
            for file, ref_file, logp, ref_logp in files:
                for path in [file, ref_file, logp, ref_logp]:
                    if path is None:
                        continue
                    if not os.path.exists(os.path.join(prefix, path)):
                        print(f'File not found: {os.path.join(prefix, path)}')
                        continue
                    
                    with open(os.path.join(prefix, path)) as f:
                        content = f.read()
                    
                    if content.strip()[-1] != ']':
                        content = f'{content}\n]'
                        with open(os.path.join(prefix, path), 'w') as f:
                            f.write(content)
                        
                        print(f'Fixed file: {os.path.join(prefix, path)}')
        
        for name, files, n in data_files:
            actual_files = []
            
            if participant_id not in results:
                results[participant_id] = defaultdict(lambda: [])
            
            if name not in results[participant_id]:
                results[participant_id][name] = []
            
            if len(results[participant_id][name]) >= model_iter and results[participant_id][name][model_iter - 1] is not None:
                continue
            
            for file, ref_file, logp, ref_logp in files:
                
                if not os.path.exists(os.path.join(prefix, file)):
                    print(f'File not found: {file}')
                    continue
                
                if ref_file is not None and not os.path.exists(os.path.join(prefix, ref_file)):
                    print(f'File not found: {ref_file}')
                    continue
                
                if logp is not None and not os.path.exists(os.path.join(prefix, logp)):
                    print(f'File not found: {logp}')
                    continue
                
                if ref_logp is not None and not os.path.exists(os.path.join(prefix, ref_logp)):
                    print(f'File not found: {ref_logp}')
                    continue
                
                if 'sorted' in file:
                    
                    # Prepare sorted_report files
                    # Example: NUM_MODELS=4 NUM_DECEPTIVE_MODELS=1 PREDICTOR_MODEL_ID=3 ONLYTHESE=1 python3 ./run_experiment_deceptivewitness.py
                    run_env = os.environ.copy()
                    run_env['NUM_MODELS'] = str(n)
                    run_env['NUM_DECEPTIVE_MODELS'] = str(1)
                    run_env['HALT_BEFORE_LOAD'] = str(1)
                    run_env['PREDICTOR_MODEL_ID'] = str(model_id)
                    run_env['ONLYTHESE'] = str(1)
                    if mix_num is not None: run_env['MIXED'] = str(1 if n == 6 else 2)
                    if 'part7' in file: run_env['PARTITION'] = '7'
                    if participant_base is not None: run_env['BASE_MODEL'] = participant_base
                    print(f'Running: num_models={n}, num_deceptive_models=1, predictor_model_id={model_id}, onlythese=1')
                    process = subprocess.Popen(['python3', './run_experiment_deceptivewitness.py'], env=run_env)
                    assert process.wait() == 0
                    
                    if ref_file is not None:
                        # Get baseline file
                        # Example: NUM_MODELS=2 NUM_DECEPTIVE_MODELS=0 PREDICTOR_MODEL_ID=3 ONLYTHESE=1 REWEIGHT=4 python3 ./run_experiment_deceptivewitness.py
                        run_env['NUM_MODELS'] = str(2)
                        run_env['NUM_DECEPTIVE_MODELS'] = str(0)
                        run_env['REWEIGHT'] = str(n)
                        print(f'Running: num_models=2, num_deceptive_models=0, reweight={n}')
                        process = subprocess.Popen(['python3', './run_experiment_deceptivewitness.py'], env=run_env)
                        assert process.wait() == 0
                
                actual_files.append((os.path.join(prefix, file), None if ref_file is None else os.path.join(prefix, ref_file)))
            
            if len(actual_files) == 0:
                print(f'No files found for {name}')
                results[participant_id][name].append(None)
                continue
            
            r2 = get_metrc(actual_files, use_counterfactual=(os.environ.get('MIXED', None) is None))
            
            if len(results[participant_id][name]) < model_iter:
                results[participant_id][name].append(r2)
            else:
                results[participant_id][name][model_iter - 1] = r2
            
            with open(output_path, 'w') as f:
                json.dump(dict(results), f, indent=2)