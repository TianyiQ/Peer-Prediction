file_path_prefix = '../datasets/'
file_collections = [
    'MMLU/test/',
    'MMLU/auxiliary_train/',
    'MMLU-PRO/data/'
]
files_ignored = ['validation.json', 'race.csv', 'obqa-full.csv', 'mc_test-full.csv', 'arc_easy-full.csv']

from typing import List, Dict, Any
import os
import json
import csv
import tqdm

all_cases: List[Dict[str, Any]] = []

# Sample json file containing 2 cases
# {
#     "question_id": {
#         "0": 0,
#         "1": 1
#     },
#     "question": {
#         "0": "The symmetric group $S_n$ has $\n\\factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of the ring 2Z.",
#         "1": "Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d\/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?"
#     },
#     "options": {
#         "0": [
#             "0",
#             "30",
#             "3",
#             "10",
#             "12",
#             "50",
#             "2",
#             "100",
#             "20",
#             "5"
#         ],
#         "1": [
#             "ST + TS is the identity map of V onto itself.",
#             "TS = 0",
#             "ST = 1",
#             "ST - TS = 0",
#             "ST = T",
#             "ST = 0",
#             "ST = TS",
#             "ST - TS is the identity map of V onto itself.",
#             "TS = T",
#             "ST = S"
#         ]
#     },
#     "answer": {
#         "0": "A",
#         "1": "H"
#     },
#     "answer_index": {
#         "0": 0,
#         "1": 7
#     },
#     "cot_content": {
#         "0": "A: Let's think step by step. A characteristic of a ring is R is $n$ if the statement $ka = 0$ for all $a\\in 2Z$ implies that $k$ is a multiple of $n$. Assume that $ka = 0$ for all $a\\in 2Z$ for some $k$. In particular $2k = 0$. Hence $k=0$ and $n=0$. The answer is (A).",
#         "1": "A: Let's think step by step. For a given polynomial $p$ we have\n\\[ST(p) = (xp(x))\u2019 = p(x) + xp\u2019(x)\\]\nand\n\\[TS(p) = xp\u2019(x).\\]\nHence \\[ST(p) - TS(p) = p(x) + xp\u2019(x) - xp\u2019(x).\\] The answer is (H)."
#     },
#     "category": {
#         "0": "math",
#         "1": "math"
#     },
#     "src": {
#         "0": "cot_lib-abstract_algebra",
#         "1": "cot_lib-college_mathematics"
#     }
# }

def add_json_cases(file_path: str, file_id: str = '', dataset: str = 'MMLUPRO') -> None:
    with open(file_path) as f:
        data = json.load(f)
    
    for key, question in tqdm.tqdm(data['question'].items()):
        all_cases.append({
            'type': data['category'][key] + '-' + file_id + '-' + dataset,
            'question': question + f'\nOptions: {str(data["options"][key])}',
            'solution': f'{data["answer"][key]}, i.e. {data["options"][key][data["answer_index"][key]]}\n\n{data["cot_content"][key]}',
            'source_path': file_path,
            'source_id': key,
            'original_source': data['src'][key],
        })

def add_csv_cases(file_path: str, file_id: str = '') -> None:
    # 6 columns: question, option1, option2, option3, option4, answer
    with open(file_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm.tqdm(enumerate(reader)):
            assert len(row) == 6, f'Expected 6 columns, got {len(row)}'
            question = row[0]
            options = row[1:5]
            answer = row[5]
            all_cases.append({
                'type': file_id,
                'question': question + f'\nOptions: {str(options)}',
                'solution': answer,
                'source_path': file_path,
                'source_id': i
            })

# Add all cases in a file to the list
def add_cases(file_path: str, file_id: str = '') -> None:
    print(f'Adding cases from {file_path}')
    if file_path.endswith('.json'):
        add_json_cases(file_path, file_id)
    elif file_path.endswith('.csv'):
        add_csv_cases(file_path, file_id)
    else:
        print(f'File type not recognized: {file_path}')


# iterate over all files in the collections
for collection in file_collections:
    for file_name in os.listdir(os.path.join(file_path_prefix, collection)):
        if file_name in files_ignored:
            continue
        add_cases(os.path.join(file_path_prefix, collection, file_name), file_name.split('.')[0])

with open('cases.json', 'w') as f:
    json.dump(all_cases, f, indent=2)