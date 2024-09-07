import json
import random
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# FIXME: parameter

# for epoch in range(20):
#     print(f"epoch: {epoch}")
negative_epoch = 19
# codegen part 
codegen_source_positive_path = f"outputs_codegen6B_1k_v3_ppl/HumanEval_samples_test_codegen-6B-multi_f32_temp0.0_test_epoch-1.jsonl"
codegen_ground_truth_prob_positive_path = f"probs_codegen_6b_1k_ground_truth_3/HumanEval_samples_test_codegen-6B-multi_f32_temp0.0_test_epoch-1.jsonl"
codegen_source_positive_sample_path = f"outputs_codegen6B_1k_v3_ppl/HumanEval_samples_test_codegen-6B-multi_f32_temp0.8_test_epoch-1.jsonl"
codegen_source_negative_path = f"outputs_codegen-6B_new_humaneval_10k_0217/HumanEval_samples_test_codegen-6B-multi_f32_temp0.0_test_epoch{negative_epoch}.jsonl"
codegen_ground_truth_prob_negative_path = f"probs_codegen_6b_new_humaneval_1k_ground_truth_3/HumanEval_samples_test_codegen-6B-multi_f32_temp0.0_test_epoch{negative_epoch}.jsonl"
codegen_source_negative_sample_path = f"outputs_codegen-6B_new_humaneval_10k_0217/HumanEval_samples_test_codegen-6B-multi_f32_temp0.8_test_epoch{negative_epoch}.jsonl"
codegen_new_dataset = []
# ---------

# codellama part 
codellama_source_positive_path = f"outputs_CodeLlama7B_10k_ppl/HumanEval_samples_test_CodeLlama-7b-hf_f32_temp0.0_test_epoch-1.jsonl"
codellama_ground_truth_prob_positive_path = f"probs_codellama_7b_10k_ground_truth_3/HumanEval_samples_test_CodeLlama-7b-hf_f32_temp0.0_test_epoch-1.jsonl"
codellama_source_positive_sample_path = f"outputs_noset_10k/HumanEval_samples_test_CodeLlama-7b-hf_f32_temp0.8_test_epoch-1.jsonl"
codellama_source_negative_path = f"outputs_CodeLlama7B_new_humaneval_10k_0217/HumanEval_samples_test_CodeLlama-7b-hf_f32_temp0.0_test_epoch{negative_epoch}.jsonl"
codellama_ground_truth_prob_negative_path = f"probs_codellama_7b_new_humaneval_10k_ground_truth_3/HumanEval_samples_test_CodeLlama-7b-hf_f32_temp0.0_test_epoch{negative_epoch}.jsonl"
codellama_negative_sample_path = f"outputs_CodeLlama7B_new_humaneval_10k_0217/HumanEval_samples_test_CodeLlama-7b-hf_f32_temp0.8_test_epoch{negative_epoch}.jsonl"
codellama_new_dataset = []

save_path = f"datasets/detect_dataset_all_varient_v3/variant_data_contamination_detection_dataset_truncate_epoch{negative_epoch}"


# ---------------------------------------------------------------------------------------------

def truncate(d, method_name):
    output = d.replace("'''", '"""')
    output = output[output.find('def '+method_name):]
    output = output[output.find('"""')+3:]
    output = output[output.find('"""\n')+4:] if '"""\n' in output else output[output.find('"""')+3:]

    return output

import re
def original_truncate_back(d, method_name):
    pred = d[:d.find('def '+method_name)]
    d = d[d.find('def '+method_name):].split('\n\n')
    s = pred + d[0] + '\n\n'
    if len(d)>1:
        for i in d[1:]:
            if (('def' not in i) or ("  def" in i)) and i and '__main__' not in i and i.strip()[:3]!="def":
                s += i + '\n\n'
            else:
                break
    return s

def truncate_back(d, method_name):
    pred = d[:d.find('def '+method_name)]
    d = d[d.find('def '+method_name):]
    line = d.split('\n')

    code = [line[0]]
    for l in line[1:]:
        if len(l.strip()) == 0:
            code.append(l)
            continue
        indent = len(l) - len(l.lstrip())
        if indent == 0:
            break
        else:
            code.append(l)

    return pred + '\n'.join(code).strip()

        
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

def get_edit_distance_distribution(gready_sample, samples, tokenizer, length = 256):
    gready_sample = gready_sample.strip()
    gs = tokenizer.encode(gready_sample, add_special_tokens=False)[:length] if length else tokenizer.encode(gready_sample, add_special_tokens=False)
    num = []
    max_length = len(gs)
    for sample in samples:
        sample = sample.strip()
        s = tokenizer.encode(sample, add_special_tokens=False)[:length] if length else tokenizer.encode(sample, add_special_tokens=False)
        num.append(levenshtein_distance(gs, s))
        max_length = max(max_length, len(s))
    return num, max_length

def calculate_ratio(numbers, alpha=1):
    count = sum(1 for num in numbers if num <= alpha)
    total = len(numbers)
    ratio = count / total if total > 0 else 0
    return ratio

def remove_cases(dataset, tokenizer):
    tasks = []
    task_ids = []
    for task in dataset:
        if task['label'] == 0:
            dist, ml = get_edit_distance_distribution(task['completion'], task['completion_sample'], tokenizer)
            peaked = calculate_ratio(dist, 2)
            if peaked <= 0.02:
                tasks.append(task)
                task_ids.append(task['task_id'])
    return tasks, task_ids

codegen_tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/codegen-6B-multi",cache_dir="/data3/public_checkpoints/huggingface_models/codegen-6b-multi", trust_remote_code=True)
codegen_tokenizer.pad_token = codegen_tokenizer.eos_token
codegen_tokenizer.padding_side = "right"
# ---------    

codellama_tokenizer = AutoTokenizer.from_pretrained(f"/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6", trust_remote_code=True)
codellama_tokenizer.pad_token = codellama_tokenizer.eos_token
codellama_tokenizer.padding_side = "right"

# ---------------------------------------------------------------------------------------------

humaneval_task_map = {}
humaneval_dataset = load_from_disk('datasets/humaneval')
for task in humaneval_dataset['test']:
    humaneval_task_map[task['task_id']] = task

# ---------------------------------------------------------------------------------------------

# def construct_dataset(source_path, ground_truth_prob_path, source_sample_path, model_name, label, new_dataset):

with open(codegen_source_positive_path, "r") as f:
    codegen_source_positive = [json.loads(line) for line in f]

codegen_ground_truth_prob_positive_task_map = {}
with open(codegen_ground_truth_prob_positive_path, "r") as f:
    codegen_ground_truth_prob_positive = [json.loads(line) for line in f]
    for task in codegen_ground_truth_prob_positive:
        codegen_ground_truth_prob_positive_task_map[task["task_id"]] = task["prob"]

codegen_source_positive_sample_task_map = {}
with open(codegen_source_positive_sample_path, "r") as f:
    codegen_source_positive_sample = [json.loads(line) for line in f]
    for task in codegen_source_positive_sample:
        if task["task_id"] not in codegen_source_positive_sample_task_map.keys():
            codegen_source_positive_sample_task_map[task["task_id"]] = []

        completion_truncate_back = truncate_back(task["completion"],humaneval_task_map[task["task_id"]]["entry_point"])
        task["completion"] = truncate(completion_truncate_back,humaneval_task_map[task["task_id"]]["entry_point"])

        codegen_source_positive_sample_task_map[task["task_id"]].append(task["completion"])

with open(codegen_source_negative_path, "r") as f:
    codegen_source_negative = [json.loads(line) for line in f]

codegen_ground_truth_prob_negative_task_map = {}
with open(codegen_ground_truth_prob_negative_path, "r") as f:
    codegen_ground_truth_prob_negative = [json.loads(line) for line in f]
    for task in codegen_ground_truth_prob_negative:
        codegen_ground_truth_prob_negative_task_map[task["task_id"]] = task["prob"]

codegen_source_negative_sample_task_map = {}
with open(codegen_source_negative_sample_path, "r") as f:
    codegen_source_negative_sample = [json.loads(line) for line in f]
    for task in codegen_source_negative_sample:
        if task["task_id"] not in codegen_source_negative_sample_task_map.keys():
            codegen_source_negative_sample_task_map[task["task_id"]] = []

        completion_truncate_back = truncate_back(task["completion"],humaneval_task_map[task["task_id"]]["entry_point"])
        task["completion"] = truncate(completion_truncate_back,humaneval_task_map[task["task_id"]]["entry_point"])
        
        codegen_source_negative_sample_task_map[task["task_id"]].append(task["completion"])

# ---------

with open(codellama_source_positive_path, "r") as f:
    codellama_source_positive = [json.loads(line) for line in f]

codellama_ground_truth_prob_positive_task_map = {}
with open(codellama_ground_truth_prob_positive_path, "r") as f:
    codellama_ground_truth_prob_positive = [json.loads(line) for line in f]
    for task in codellama_ground_truth_prob_positive:
        codellama_ground_truth_prob_positive_task_map[task["task_id"]] = task["prob"]

codellama_source_positive_sample_task_map = {}
with open(codellama_source_positive_sample_path, "r") as f:
    codellama_source_positive_sample = [json.loads(line) for line in f]
    for task in codellama_source_positive_sample:
        if task["task_id"] not in codellama_source_positive_sample_task_map.keys():
            codellama_source_positive_sample_task_map[task["task_id"]] = []

        completion_truncate_back = truncate_back(task["completion"],humaneval_task_map[task["task_id"]]["entry_point"])
        task["completion"] = truncate(completion_truncate_back,humaneval_task_map[task["task_id"]]["entry_point"])
        
        codellama_source_positive_sample_task_map[task["task_id"]].append(task["completion"])

with open(codellama_source_negative_path, "r") as f:
    codellama_source_negative = [json.loads(line) for line in f]

codellama_ground_truth_prob_negative_task_map = {}
with open(codellama_ground_truth_prob_negative_path, "r") as f:
    codellama_ground_truth_prob_negative = [json.loads(line) for line in f]
    for task in codellama_ground_truth_prob_negative:
        codellama_ground_truth_prob_negative_task_map[task["task_id"]] = task["prob"]

codellama_source_negative_sample_task_map = {}
with open(codellama_negative_sample_path, "r") as f:
    codellama_negative_sample = [json.loads(line) for line in f]
    for task in codellama_negative_sample:
        if task["task_id"] not in codellama_source_negative_sample_task_map.keys():
            codellama_source_negative_sample_task_map[task["task_id"]] = []

        completion_truncate_back = truncate_back(task["completion"],humaneval_task_map[task["task_id"]]["entry_point"])
        task["completion"] = truncate(completion_truncate_back,humaneval_task_map[task["task_id"]]["entry_point"])
        
        codellama_source_negative_sample_task_map[task["task_id"]].append(task["completion"])

# ---------------------------------------------------------------------------------------------

for i in range(len(codegen_source_positive)):
    task = codegen_source_positive[i]
    task['model_name'] = "codegen-6B-multi"
    task["label"] = 0

    completion_truncate_back = truncate_back(task["completion"],humaneval_task_map[task["task_id"]]["entry_point"])
    task["completion"] = truncate(completion_truncate_back,humaneval_task_map[task["task_id"]]["entry_point"])

    task["prob"] = json.dumps(task["prob"])
    task["leaked_data"] = humaneval_task_map[task["task_id"]]["canonical_solution"]
    task["ground_truth_prob"] = json.dumps(codegen_ground_truth_prob_positive_task_map[task["task_id"]])
    task["completion_sample"] = codegen_source_positive_sample_task_map[task["task_id"]]

    codegen_new_dataset.append(task)
codegen_new_dataset, remove_ids = remove_cases(codegen_new_dataset, codegen_tokenizer)
print("len codegen -1:", len(codegen_new_dataset))

for i in range(len(codegen_source_negative)):
    task = codegen_source_negative[i]

    if task['task_id'] not in remove_ids:
        continue

    task['model_name'] = "codegen-6B-multi"    
    task["label"] = 1

    completion_truncate_back = truncate_back(task["completion"],humaneval_task_map[task["task_id"]]["entry_point"])
    task["completion"] = truncate(completion_truncate_back,humaneval_task_map[task["task_id"]]["entry_point"])

    task["prob"] = json.dumps(task["prob"])
    task["leaked_data"] = humaneval_task_map[task["task_id"]]["canonical_solution"]
    task["ground_truth_prob"] = json.dumps(codegen_ground_truth_prob_negative_task_map[task["task_id"]])
    task["completion_sample"] = codegen_source_negative_sample_task_map[task["task_id"]]

    codegen_new_dataset.append(task)
print("len codegen 0:", len(codegen_new_dataset))

# ---------

for i in range(len(codellama_source_positive)):
    task = codellama_source_positive[i]
    task['model_name'] = "CodeLlama-7b"
    task["label"] = 0

    completion_truncate_back = truncate_back(task["completion"],humaneval_task_map[task["task_id"]]["entry_point"])
    task["completion"] = truncate(completion_truncate_back,humaneval_task_map[task["task_id"]]["entry_point"])

    task["prob"] = json.dumps(task["prob"])
    task["leaked_data"] = humaneval_task_map[task["task_id"]]["canonical_solution"]
    ground_truth_prob = codellama_ground_truth_prob_positive_task_map[task["task_id"]]
    task["ground_truth_prob"] = json.dumps(ground_truth_prob)
    task["completion_sample"] = codellama_source_positive_sample_task_map[task["task_id"]]

    codellama_new_dataset.append(task)
codellama_new_dataset, remove_ids = remove_cases(codellama_new_dataset, codellama_tokenizer)
print("len codellama -1:", len(codellama_new_dataset))

for i in range(len(codellama_source_negative)):
    task = codellama_source_negative[i]

    if task['task_id'] not in remove_ids:
        continue
    
    task['model_name'] = "CodeLlama-7b"    
    task["label"] = 1

    completion_truncate_back = truncate_back(task["completion"],humaneval_task_map[task["task_id"]]["entry_point"])
    task["completion"] = truncate(completion_truncate_back,humaneval_task_map[task["task_id"]]["entry_point"])
    
    task["prob"] = json.dumps(task["prob"])
    task["leaked_data"] = humaneval_task_map[task["task_id"]]["canonical_solution"]
    ground_truth_prob = codellama_ground_truth_prob_negative_task_map[task["task_id"]]
    task["ground_truth_prob"] = json.dumps(ground_truth_prob)
    task["completion_sample"] = codellama_source_negative_sample_task_map[task["task_id"]]

    codellama_new_dataset.append(task)

print("len codellama 0:", len(codellama_new_dataset))
# ---------------------------------------------------------------------------------------------


#codegen_new_dataset.extend(codellama_new_dataset)
new_dataset = codellama_new_dataset

random.shuffle(new_dataset)

for i in range(len(new_dataset)):   
    new_dataset[i]["id"] = i

new_dataset = Dataset.from_list(new_dataset)
new_dataset = DatasetDict({'test': new_dataset})

new_dataset.save_to_disk(save_path)


