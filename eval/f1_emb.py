import json
import torch
import random
import os
from datasets import load_from_disk
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

from transformers import AutoModelForCausalLM

from sentence_transformers import SentenceTransformer


# base_model = AutoModelForCausalLM.from_pretrained(
#             f"/home/jiangxue/LLMs/CodeLlama-7b-hf",
#             low_cpu_mem_usage=True,
#             revision="float32",
#             torch_dtype=torch.float32,
#         ).cuda()

def compute_f1score(TP, FP, FN):
    # Compute precision and recall
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1


def get_embedding(model, text):
    query_embedding = torch.tensor(model.encode(text)).cuda().unsqueeze(0).unsqueeze(0)
    return query_embedding


def get_programs(r_path):
    dataset = [json.loads(l) for l in open(r_path, "r")]
    programs = [each["text"] for each in dataset]
    return programs

def work():
    languages = ["python", "c", "js"]
    model = SentenceTransformer("/data3/public_checkpoints/huggingface_models/multi-qa-MiniLM-L6-cos-v1")

    threshold = 0.6

    original_programs = get_programs("/home/zhangxuanming/DataLeakage_v2/eval/data/test/HumanEval.jsonl")
    
    dataset = load_from_disk('/home/zhangxuanming/DataLeakage_v2/datasets/humaneval')['test']
    original_programs = [task['canonical_solution'] for task in dataset]
    #dataset = load_from_disk('/home/zhangxuanming/DataLeakage_v2/datasets/gsm8k_main')['test']
    #original_programs = [task['answer'] for task in dataset]

    origin_embs = []

    for i in range(len(original_programs)):
        origin_embs.append(torch.mean(get_embedding(model, original_programs[i])[0], dim=0))

    FP = 0

    dataset = load_from_disk("/home/zhangxuanming/DataLeakage_v2/datasets/detect_dataset_all_v3/original_data_contamination_detection_dataset_truncate_epoch2")['test']

    res = []
    for x in range(len(dataset)):
        task = dataset[x]
        emb = torch.mean(get_embedding(model, task['completion'])[0], dim=0)
        # i = int(task['task_id'].split("/")[-1])
        task_id = task['task_id']
        if '/' in task_id:
            i = int(task_id.split('/')[-1])
        else:
            i = int(task_id)
        # if torch.cosine_similarity(origin_embs[i], emb, dim=0).item() > threshold:
        #     res.append(False)  # 余弦相似度大说明不泄露
        # else:
        #     res.append(True)
        res.append(torch.cosine_similarity(origin_embs[i], emb, dim=0).item())
    return res
# cnt = 0
# rand_embs = random.sample(origin_embs, 15)

# for i in range(len(rand_embs)):
#     if cnt >= 100:
#         break
#     for j in range(i + 1, len(rand_embs)):
#         cnt += 1
#         if torch.cosine_similarity(rand_embs[i], rand_embs[j], dim=0).item() > threshold:
#             FP += 1
#         if cnt >= 100:
#             break
#
# print(FP)
#
# te_f1 = compute_f1score(100, FP, 0)
# print(f"Test set F1 score: {te_f1}")

# for language in languages:
#     rephrased_programs = get_programs(f"data/rephrase/humaneval_{language}.jsonl")
#
#     rephrase_embs = []
#
#     for i in range(len(original_programs)):
#         rephrase_embs.append(torch.mean(get_embedding(model, rephrased_programs[i])[0], dim=0))
#
#     re_TP = 0
#     re_FN = 0
#
#     for i in range(len(origin_embs)):
#         if torch.cosine_similarity(origin_embs[i], rephrase_embs[i], dim=0).item() > threshold:
#             re_TP += 1
#         else:
#             re_FN += 1
#
#     print(re_TP)
#     print(re_FN)
#
#     re_f1 = compute_f1score(re_TP, FP, re_FN)
#
#     print(f"Rephrase {language} F1 score: {re_f1}")