import json
from datasets import load_from_disk

MAX_NUM = 1e9

def levenshtein_distance(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1

    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]

def strip_code(sample):
    return sample.strip().split('\n\n\n')[0] if '\n\n\n' in sample else sample.strip().split('```')[0]

def tokenize_code(sample, tokenizer, length):
    return tokenizer.encode(sample)[:length] if length else tokenizer.encode(sample)

def get_edit_distance_distribution_star(samples, gready_sample, tokenizer, length = 100):
    gready_sample = strip_code(gready_sample)
    gs = tokenize_code(gready_sample, tokenizer, length)
    num = []
    max_length = len(gs)
    for sample in samples:
        sample = strip_code(sample)
        # add rule to avid empty output
        if gready_sample=='' and sample=='':
            num.append(MAX_NUM)
            continue
        s = tokenize_code(sample, tokenizer, length)
        num.append(levenshtein_distance(gs, s))
        max_length = max(max_length, len(s))
    return num, max_length

def calculate_ratio(numbers, alpha=1):
    count = sum(1 for num in numbers if num <= alpha)
    total = len(numbers)
    ratio = count / total if total > 0 else 0
    return ratio

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default='0.05')
parser.add_argument('--xi', type=float, default='0.01')
parser.add_argument('--input_path', type=str, default='datasets/detect_dataset_all_varient_v3/variant_data_contamination_detection_dataset_truncate_epoch2') #variant

parser.add_argument('--model', type=str, default='CodeLLaMa-Ins-34B')

args = parser.parse_args()

if __name__ == '__main__':
    if 'gpt' in args.model:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/home/jiangxue/LLMs/CodeLlama-7b-hf")
        
    dataset = load_from_disk(args.input_path)['test']
    
    results=[]
    for task in dataset:
        # task['samples'] temperature = 0.8 num = 50
        # task['gready_sample'] temperature = 0 num = 1
        dist, ml = get_edit_distance_distribution_star(task['completion_sample'], task['completion'], tokenizer)
        peaked = calculate_ratio(dist, args.alpha*ml) 
        results.append((task["task_id"],task, peaked))
    
    # print(args.xi, sum([i>args.xi for i in results])/len(results))

    task_id_map = {}
    for task_id, task, peaked in results:
        if task["model_name"] == "CodeLlama-7b":
            task['peaked'] = peaked
            if task_id not in task_id_map:
                task_id_map[task_id] = []
            task_id_map[task_id].append(task)

    peak_sort = []
    for task_id in task_id_map.keys():
        peak_sort.append((task_id, abs(task_id_map[task_id][0]['peaked']-task_id_map[task_id][1]['peaked'])))

    peak_sort.sort(key=lambda x: x[1], reverse=True)

    # find the top 5 peak from task_id_map, save to file
    top_5 = {}
    for i in range(50):
        task_id = peak_sort[i][0]
        print(task_id)
        for task in task_id_map[task_id]:
            print(task['peaked'], task['model_name'], task['completion'], task['completion_sample'])
        top_5[task_id] = task_id_map[task_id]

    with open('top_5.jsonl', 'w') as f:
        for task_id in top_5:
            for task in top_5[task_id]:
                f.write(json.dumps(task) + '\n')


with open("top_5_original.jsonl", "r") as f:
    top_5_original = [json.loads(line) for line in f]


    
