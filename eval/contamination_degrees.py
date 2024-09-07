from datasets import load_from_disk, load_dataset
import json,os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

def get_edit_distance_distribution(samples, gready_sample, tokenizer, length = 100):
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

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

def evaluate_classification(y_true, y_pred, y_pred_prob=None):
    metrics = {
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    
    # 如果提供了预测概率，则计算AUC值
    if y_pred_prob is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_prob)
    
    return metrics

# Dataset({
#     features: ['task_id', 'prompt', 'completion', 'prob', 'label', 'leaked_data', 'ground_truth_prob', 'completion_sample', 'id'],
#     num_rows: 656
# })

from transformers import AutoTokenizer
codellama_tokenizer = AutoTokenizer.from_pretrained(f"/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6")
#codegen_tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/codegen-6B-multi")
# Llama_tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-7b")
# mistral_tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-7B-v0.1", trust_remote_code=True)

import math
import numpy as np

def calc_ppl(probs):
    all_prob = []
    for prob in probs:
        v = list(prob.values())
        all_prob.append(v[0])
    probs = all_prob
    total_log_prob = sum([-math.log(prob) for prob in probs])
    perplexity = math.exp(total_log_prob / len(probs)) if len(probs) else math.exp(0)
    return -perplexity #< 1.5  # 困惑度大说明没看过样本，未泄露为1

def min_k(probs):
    all_prob = []
    pred = {}
    for prob in probs:
        v = list(prob.values())
        all_prob.append(v[0])
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  # min-k，k的取值选择多少多少
        k_length = int(len(all_prob) * ratio)
        if k_length:
            topk_prob = np.sort(all_prob)[:k_length]
            pred[f"Min_{ratio * 100}% Prob"] = np.mean(topk_prob).item()
        else:
            pred[f"Min_{ratio * 100}% Prob"] = 0
    # 选择k的取值，本处k=0.05
    k = 0.05
    # print(pred[f"Min_{k * 100}% Prob"])
    return pred[f"Min_{k * 100}% Prob"]# > epsilon  # 大于epsilon认为泄露，返回1

def generate_ngrams_token(text, n):
    text = codellama_tokenizer.tokenize(text)
    if len(text) < n:
        return [" ".join(text)]
    ngrams = [" ".join(text[i:i+n]) for i in range(len(text)-n+1)]
    return ngrams

def generate_ngrams_char(text, n):
    if len(text) < n:
        return [text]
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    return ngrams

def detect_duplicates(text1, text2, level="token_level", n=13):
    """
    Detect duplicate n-grams between two texts.

    :param text1: The first input text string.
    :param text2: The second input text string.
    :param n: The number of characters in each n-gram.
    :return: A set of duplicate n-grams found in both texts.
    """

    if level=="token_level":
        ngrams1 = set(generate_ngrams_token(text1, n))
        ngrams2 = set(generate_ngrams_token(text2, n))
    else:
        ngrams1 = set(generate_ngrams_char(text1, n))
        ngrams2 = set(generate_ngrams_char(text2, n))

    # Find the intersection of the two sets
    duplicates = ngrams1.intersection(ngrams2)
    if len(duplicates) > 0:
        return 1
    else:
        return 0

def n_gram_detection(dataset, level="token_level", n=13):
    y_true = []
    y_pred = []
    for task in dataset:
        duplicates = detect_duplicates(task['completion'], task['leaked_data'], level, n)
        y_true.append(task['label'])
        y_pred.append(duplicates)
    return y_true, y_pred

import time
# import multiprocessing

# def compute_edit_distance(args):
#     def unsafe_execute():
#         gs, sample, s = args
#         sample = sample.strip()
#         ld = levenshtein_distance(gs, s)
#         result.append([sample, ld , len(s)])

#     manager = multiprocessing.Manager()
#     result = manager.list()

#     p = multiprocessing.Process(target=unsafe_execute)
#     p.start()
#     p.join(timeout=2)
#     if p.is_alive():
#         p.kill()
    
#     # if not result:
#     #     result.append([args[1], 100, 0])
    
#     return result[0]

# from concurrent.futures import as_completed, ProcessPoolExecutor

# def get_edit_distance_distribution(samples, greedy_sample, tokenizer, length=100):
#     greedy_sample = greedy_sample.strip()
#     gs = tokenizer.encode(greedy_sample, add_special_tokens=False)[:length] if length else tokenizer.encode(greedy_sample, add_special_tokens=False)
#     num = []
#     max_length = len(gs)
    
#     # 使用ProcessPoolExecutor并行计算
#     with ProcessPoolExecutor() as executor:
#         futures = []
#         existed_completion = set()
#         results = dict()
#         for sample in samples:
#             sample = sample.strip()
#             if sample in existed_completion:
#                 continue
#             existed_completion.add(sample)
#             s = tokenizer.encode(sample, add_special_tokens=False)[:length] if length else tokenizer.encode(sample, add_special_tokens=False)
#             args = (gs, sample, s)
#             future = executor.submit(compute_edit_distance, args)
#             futures.append(future)
    
#         # 更新num和max_length
#         for future in as_completed(futures):
#             result = future.result()
#             results[result[0]] = result[1]

#     for sample in samples:
#         sample = sample.strip()
#         num.append(results[sample])
#         # max_length = max(max_length, results[sample][1])
#     return num, max_length

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def strip_code(sample):
    return sample.strip().split('\n\n\n')[0] if '\n\n\n' in sample else sample.strip().split('```')[0]

def tokenize_code(sample, tokenizer, length):
    return tokenizer.encode(sample)[:length] if length else tokenizer.encode(sample)

def process_batch(batch, gready_sample, tokenizer, length):
    results = []
    gready_sample = strip_code(gready_sample)
    gs = tokenize_code(gready_sample, tokenizer, length)
    max_length = len(gs)
    for sample in batch:
        sample = strip_code(sample)
        if gready_sample == '' and sample == '':
            results.append(MAX_NUM)
            continue
        s = tokenize_code(sample, tokenizer, length)
        results.append(levenshtein_distance(gs, s))
        max_length = max(max_length, len(s))
    return results, max_length


def get_edit_distance_distribution_star(samples, gready_sample, tokenizer, length=100, batch_size=50):
    num = []
    max_length = 0

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            future = executor.submit(process_batch, batch, gready_sample, tokenizer, length)
            futures.append(future)

        for future in as_completed(futures):
            result, batch_max_length = future.result()
            num.extend(result)
            max_length = max(max_length, batch_max_length)

    return num, max_length

alpha = 0.05
xi = 0.02
Accuracy = []
Precision = []
Recall = []
F1Score = []
AUC = []
results = []
GroundTruths = []
fillkey = False
#ToDo
#for epoch in range(0, 1):
for epoch in [2]:
    path = f'/home/zhangxuanming/DataLeakage_v2/datasets/detect_dataset_all_v3/original_data_contamination_detection_dataset_truncate_epoch{epoch}'

    if not os.path.exists(path):
        Accuracy.append(metric['Accuracy'])
        Precision.append(metric['Precision'])
        Recall.append(metric['Recall'])
        F1Score.append(metric['F1 Score'])
        AUC.append(metric['AUC']) 
        fillkey = True
        continue
    else:
        dataset = load_from_disk(path)['test']
        result = []
        GroundTruth = []
        GroundTruth, result =  n_gram_detection(dataset, level="token_level", n=13)
        # 记录开始时间
        start_time = time.time()
        for task in dataset:
            if task['model_name'] == "CodeLlama-7b":
                tokenizer = codellama_tokenizer
            else:
                tokenizer = codegen_tokenizer
            
            #emb = torch.mean(get_embedding(model, task['completion'])[0], dim=0)
            #tem = torch.mean(get_embedding(model, task['leaked_data'])[0], dim=0)
            #result.append(torch.cosine_similarity(tem, emb, dim=0).item())
            #Todo
            #GroundTruth.append(task['label'])
                
            dist, ml = get_edit_distance_distribution_star(task['completion_sample'], task['completion'], tokenizer)
            peaked = calculate_ratio(dist, alpha*ml) 
            result.append(peaked)
            GroundTruth.append(task['label'])
            
            task_prob = json.loads(task["ground_truth_prob"])
            temps = []
            #ToDo
            #result.append(min_k(probs=task_prob)) #  > 0.003
            #print(task)
            #ToDo
            result.append(calc_ppl(probs=task_prob))
            GroundTruth.append(task['label'])
        
        results.append(result)
        GroundTruths.append(GroundTruth)
    #print(f'epoch {epoch} finished')
    #print(result)
    metric = evaluate_classification(GroundTruth, [i>xi for i in result], result)
    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    run_time = end_time - start_time
    print(f"程序运行时间：{run_time}秒")
    #metric = evaluate_classification(GroundTruth, result, result)
    if fillkey:
        Accuracy[-1] = (Accuracy[-1] + metric['Accuracy'])/2
        Precision[-1] = (Precision[-1] + metric['Precision'])/2
        Recall[-1] = (Recall[-1] + metric['Recall'])/2
        F1Score[-1] = (F1Score[-1] + metric['F1 Score'])/2
        AUC[-1] = (AUC[-1] + metric['AUC'])/2
        fillkey = False
    Accuracy.append(metric['Accuracy'])
    Precision.append(metric['Precision'])
    Recall.append(metric['Recall'])
    F1Score.append(metric['F1 Score'])
    AUC.append(metric['AUC']) 
print(f'Accuracy = {Accuracy}')
print(f'Precision = {Precision}')
print(f'Recall = {Recall}')
print(f'F1Score = {F1Score}')
print(f'AUC = {AUC}')

