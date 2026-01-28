files = [
    # 'responses-8B-9-untrunc.json',
    # 'responses-8B-2.json',
    # 'responses-8B-3.json',
    'responses-8B-1-hyperdeceptive.json',
    # 'responses-8B-2-deceptive.json',
    # 'responses-8B-3-deceptive.json',
    # 'responses-8B-4-deceptive.json',
    # 'responses-8B-5-deceptive.json',
    # 'responses-8B-1-untrunc-hyperdeceptive.json',
    # 'responses-8B-3-untrunc-hyperdeceptive.json',
]

out_files = [
    'responses-8B-1-hyperdeceptive-XXX.json',
    # 'responses-8B-9.json',
    # 'responses-8B-1-trunc.json',
    # 'responses-8B-2-trunc.json',
    # 'responses-8B-3-trunc.json',
    # 'responses-8B-1-trunc-deceptive.json',
    # 'responses-8B-2-trunc-deceptive.json',
    # 'responses-8B-3-trunc-deceptive.json',
    # 'responses-8B-4-trunc-deceptive.json',
    # 'responses-8B-5-trunc-deceptive.json',
    # 'responses-8B-1-hyperdeceptive.json',
    # 'responses-8B-3-hyperdeceptive.json',
]

from transformers import AutoTokenizer
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
counter = Counter()

def truncate_num_tokens(text: str, max_num_tokens: int) -> str:
    tokens = tokenizer(text, return_tensors='pt', return_token_type_ids=False, return_attention_mask=False)['input_ids']
    if tokens.size(1) > max_num_tokens:
        # print(f"Truncating text from {tokens.size(1)} to {max_num_tokens} tokens")
        # print(f"Original text: {text}\n\n\n\n\n")
        counter[tokens.size(1)] += 1
        return tokenizer.decode(tokens[0, :max_num_tokens])
    return text

import json
from tqdm import tqdm

for file, out_file in zip(files, out_files):
    print(file, out_file)
    with open(file, 'r') as f:
        data = json.load(f)
    
    print(len(data))
    for i, d in tqdm(enumerate(data)):
        for j, response in enumerate(d['reports']):
            d['reports'][j] = truncate_num_tokens(d['reports'][j], 512)
        data[i] = d
    
    # with open(out_file, 'w') as f:
    #     json.dump(data, f, indent=4)

print(sorted(counter.items(), key=lambda x: x[0]))
print(sum(counter.values()))