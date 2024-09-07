import json
import random
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# FIXME: parameter
negative_epoch = 2

# llama part 
llama_source_positive_path = f"outputs_llama_red_pajama_data_100K_gsm8k_reformat_1k_ppl/GSM8K_samples_test_Llama-2-7b-hf_f32_temp0.0_test_epoch-1.jsonl"
llama_ground_truth_prob_positive_path = f"probs_Llama_2_7b_gsm8k_1k_ground_truth/GSM8K_samples_test_Llama-2-7b-hf_f32_temp0.0_test_epoch-1.jsonl"
llama_source_positive_sample_path = f"outputs_llama_red_pajama_data_100K_gsm8k_reformat_1k_ppl/GSM8K_samples_test_Llama-2-7b-hf_f32_temp0.8_test_epoch-1.jsonl"
llama_source_negative_path = f"outputs_llama_red_pajama_data_100K_gsm8k_reformat_1k_ppl/GSM8K_samples_test_Llama-2-7b-hf_f32_temp0.0_test_epoch{negative_epoch}.jsonl"
llama_ground_truth_prob_negative_path = f"probs_Llama_2_7b_gsm8k_1k_ground_truth/GSM8K_samples_test_Llama-2-7b-hf_f32_temp0.0_test_epoch{negative_epoch}.jsonl"
llama_source_negative_sample_path = f"outputs_llama_red_pajama_data_100K_gsm8k_reformat_1k_ppl/GSM8K_samples_test_Llama-2-7b-hf_f32_temp0.8_test_epoch{negative_epoch}.jsonl"
llama_new_dataset = []

# ---------

# mistral part 
mistral_source_positive_path = f"outputs_Mistral_red_pajama_data_100K_gsm8k_reformat_1k_ppl/GSM8K_samples_test_Mistral-7B-v0.1_f32_temp0.0_test_epoch-1.jsonl"
mistral_ground_truth_prob_positive_path = f"probs_Mistral_7B_gsm8k_1k_ground_truth/GSM8K_samples_test_Mistral-7B-v0.1_f32_temp0.0_test_epoch-1.jsonl"
mistral_source_positive_sample_path = f"outputs_Mistral_red_pajama_data_100K_gsm8k_reformat_1k_ppl/GSM8K_samples_test_Mistral-7B-v0.1_f32_temp0.8_test_epoch-1.jsonl"
mistral_source_negative_path = f"outputs_Mistral_red_pajama_data_100K_gsm8k_reformat_1k_ppl/GSM8K_samples_test_Mistral-7B-v0.1_f32_temp0.0_test_epoch{negative_epoch}.jsonl"
mistral_ground_truth_prob_negative_path = f"probs_Mistral_7B_gsm8k_1k_ground_truth/GSM8K_samples_test_Mistral-7B-v0.1_f32_temp0.0_test_epoch{negative_epoch}.jsonl"
mistral_negative_sample_path = f"outputs_Mistral_red_pajama_data_100K_gsm8k_reformat_1k_ppl/GSM8K_samples_test_Mistral-7B-v0.1_f32_temp0.8_test_epoch{negative_epoch}.jsonl"
mistral_new_dataset = []

save_path = f"datasets/original_gsm8k_data_contamination_detection_dataset_epoch{negative_epoch}"

# ---------------------------------------------------------------------------------------------

gsm8k_task_map = {}
gsm8k_dataset = load_from_disk('datasets/test_gsm8k_198')
for task in gsm8k_dataset:
    gsm8k_task_map[task['task_id']] = task

# ---------------------------------------------------------------------------------------------
    
import re
def remove_special_tokens(text):
    text = re.sub(r'^(<s>)*\s*', '', text)
    if "</s>" in text:
        text = text[: text.find("</s>")]
    return text

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

def get_edit_distance_distribution(gready_sample, samples, tokenizer, length = 100):
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

# ---------------------------------------------------------------------------------------------
llama_tokenizer = AutoTokenizer.from_pretrained(f"/home/jiangxue/LLMs/Llama-2-7b-hf", trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"
# ---------    
mistral_tokenizer = AutoTokenizer.from_pretrained(f"/home/dongyh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24", trust_remote_code=True)
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
mistral_tokenizer.padding_side = "right"


# ---------------------------------------------------------------------------------------------

with open(llama_source_positive_path, "r") as f:
    llama_source_positive = [json.loads(line) for line in f]

llama_ground_truth_prob_positive_task_map = {}
with open(llama_ground_truth_prob_positive_path, "r") as f:
    llama_ground_truth_prob_positive = [json.loads(line) for line in f]
    for task in llama_ground_truth_prob_positive:
        llama_ground_truth_prob_positive_task_map[task["task_id"]] = task["prob"]

llama_source_positive_sample_task_map = {}
with open(llama_source_positive_sample_path, "r") as f:
    llama_source_positive_sample = [json.loads(line) for line in f]
    for task in llama_source_positive_sample:
        if task["task_id"] not in llama_source_positive_sample_task_map.keys():
            llama_source_positive_sample_task_map[task["task_id"]] = []
        task["completion"] = task["completion"][len(task["prompt"]):]
        llama_source_positive_sample_task_map[task["task_id"]].append(task["completion"])

with open(llama_source_negative_path, "r") as f:
    llama_source_negative = [json.loads(line) for line in f]

llama_ground_truth_prob_negative_task_map = {}
with open(llama_ground_truth_prob_negative_path, "r") as f:
    llama_ground_truth_prob_negative = [json.loads(line) for line in f]
    for task in llama_ground_truth_prob_negative:
        llama_ground_truth_prob_negative_task_map[task["task_id"]] = task["prob"]

llama_source_negative_sample_task_map = {}
with open(llama_source_negative_sample_path, "r") as f:
    llama_source_negative_sample = [json.loads(line) for line in f]
    for task in llama_source_negative_sample:
        if task["task_id"] not in llama_source_negative_sample_task_map.keys():
            llama_source_negative_sample_task_map[task["task_id"]] = []
        task["completion"] = task["completion"][len(task["prompt"]):]
        llama_source_negative_sample_task_map[task["task_id"]].append(task["completion"])

# ---------

with open(mistral_source_positive_path, "r") as f:
    mistral_source_positive = [json.loads(line) for line in f]

mistral_ground_truth_prob_positive_task_map = {}
with open(mistral_ground_truth_prob_positive_path, "r") as f:
    mistral_ground_truth_prob_positive = [json.loads(line) for line in f]
    for task in mistral_ground_truth_prob_positive:
        mistral_ground_truth_prob_positive_task_map[task["task_id"]] = task["prob"]

mistral_source_positive_sample_task_map = {}
with open(mistral_source_positive_sample_path, "r") as f:
    mistral_source_positive_sample = [json.loads(line) for line in f]
    for task in mistral_source_positive_sample:
        if task["task_id"] not in mistral_source_positive_sample_task_map.keys():
            mistral_source_positive_sample_task_map[task["task_id"]] = []
        task["completion"] = remove_special_tokens(task["completion"])
        task["completion"] = task["completion"][len(task["prompt"]):]
        mistral_source_positive_sample_task_map[task["task_id"]].append(task["completion"])

with open(mistral_source_negative_path, "r") as f:
    mistral_source_negative = [json.loads(line) for line in f]

mistral_ground_truth_prob_negative_task_map = {}
with open(mistral_ground_truth_prob_negative_path, "r") as f:
    mistral_ground_truth_prob_negative = [json.loads(line) for line in f]
    for task in mistral_ground_truth_prob_negative:
        mistral_ground_truth_prob_negative_task_map[task["task_id"]] = task["prob"]

mistral_source_negative_sample_task_map = {}
with open(mistral_negative_sample_path, "r") as f:
    mistral_negative_sample = [json.loads(line) for line in f]
    for task in mistral_negative_sample:
        if task["task_id"] not in mistral_source_negative_sample_task_map.keys():
            mistral_source_negative_sample_task_map[task["task_id"]] = []
        task["completion"] = remove_special_tokens(task["completion"])
        task["completion"] = task["completion"][len(task["prompt"]):]
        mistral_source_negative_sample_task_map[task["task_id"]].append(task["completion"])

# ---------------------------------------------------------------------------------------------

for i in range(len(llama_source_positive)):
    task = llama_source_positive[i]
    task['model_name'] = "Llama-2-7b"
    task["label"] = 0
    task["completion"] = task["completion"][len(task["prompt"]):]
    task["prob"] = json.dumps(task["prob"])
    task["leaked_data"] = gsm8k_task_map[task["task_id"]]["canonical_solution"]
    task["ground_truth_prob"] = json.dumps(llama_ground_truth_prob_positive_task_map[task["task_id"]])
    task["completion_sample"] = llama_source_positive_sample_task_map[task["task_id"]]
    llama_new_dataset.append(task)

llama_new_dataset, remove_ids = remove_cases(llama_new_dataset, llama_tokenizer)
print("len llama -1:", len(llama_new_dataset))


for i in range(len(llama_source_negative)):
    task = llama_source_negative[i]
    if task['task_id'] not in remove_ids:
        continue    
    task['model_name'] = "Llama-2-7b"    
    task["label"] = 1
    task["completion"] = task["completion"][len(task["prompt"]):]
    task["prob"] = json.dumps(task["prob"])
    task["leaked_data"] = gsm8k_task_map[task["task_id"]]["canonical_solution"]
    task["ground_truth_prob"] = json.dumps(llama_ground_truth_prob_negative_task_map[task["task_id"]])
    task["completion_sample"] = llama_source_negative_sample_task_map[task["task_id"]]
    llama_new_dataset.append(task)

print("len llama 0:", len(llama_new_dataset))

# ---------

for i in range(len(mistral_source_positive)):
    task = mistral_source_positive[i]
    task['model_name'] = "Mistral-7B-v0.1"
    task["label"] = 0
    task["completion"] = remove_special_tokens(task["completion"])
    task["completion"] = task["completion"][len(task["prompt"]):]
    task["prob"] = json.dumps(task["prob"])
    task["leaked_data"] = gsm8k_task_map[task["task_id"]]["canonical_solution"]
    ground_truth_prob = mistral_ground_truth_prob_positive_task_map[task["task_id"]]
    task["ground_truth_prob"] = json.dumps(ground_truth_prob)
    task["completion_sample"] = mistral_source_positive_sample_task_map[task["task_id"]]
    mistral_new_dataset.append(task)

mistral_new_dataset, remove_ids = remove_cases(mistral_new_dataset, llama_tokenizer)
print("len mistral -1:", len(mistral_new_dataset))

for i in range(len(mistral_source_negative)):
    task = mistral_source_negative[i]
    if task['task_id'] not in remove_ids:
        continue    
    task['model_name'] = "Mistral-7B-v0.1"    
    task["label"] = 1
    task["completion"] = remove_special_tokens(task["completion"])
    task["completion"] = task["completion"][len(task["prompt"]):]
    task["prob"] = json.dumps(task["prob"])
    task["leaked_data"] = gsm8k_task_map[task["task_id"]]["canonical_solution"]
    ground_truth_prob = mistral_ground_truth_prob_negative_task_map[task["task_id"]]
    task["ground_truth_prob"] = json.dumps(ground_truth_prob)
    task["completion_sample"] = mistral_source_negative_sample_task_map[task["task_id"]]
    mistral_new_dataset.append(task)

print("len mistral 0:", len(mistral_new_dataset))
# ---------------------------------------------------------------------------------------------


llama_new_dataset.extend(mistral_new_dataset)
new_dataset = llama_new_dataset

random.shuffle(new_dataset)

for i in range(len(new_dataset)):   
    new_dataset[i]["id"] = i

new_dataset = Dataset.from_list(new_dataset)
new_dataset = DatasetDict({'test': new_dataset})

new_dataset.save_to_disk(save_path)


