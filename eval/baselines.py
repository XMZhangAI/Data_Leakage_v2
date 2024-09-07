from datasets import load_from_disk
import json
#import latesteval
#import f1_llm
import numpy as np
import f1_emb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

dataset = load_from_disk("/home/zhangxuanming/DataLeakage_v2/datasets/detect_dataset_all_v3/original_data_contamination_detection_dataset_truncate_epoch2")['test']
#/home/dongyh/DataLeakage/datasets/original_gsm8k_data_contamination_detection_dataset_epoch2_only_llama

# Dataset({
#     features: ['task_id', 'prompt', 'completion', 'prob', 'label', 'leaked_data', 'id'],
#     num_rows: 328
# })

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

def min_k(epsilon, probs):
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
    k = 0.6
    #print(pred[f"Min_{k * 100}% Prob"])
    return pred[f"Min_{k * 100}% Prob"]# > epsilon  # 大于epsilon认为泄露，返回1


if __name__ == "__main__":
    acc1 = 0
    acc2 = 0
    acc3 = 0
    acc4 = 0
    #for task in dataset:
        #print(task['id'])  # id in this dataset
        #print(task['task_id'])  # humaneval id
        #print(task['prompt'] + task['completion'])
        #print(task['leaked_data'])
        #print(json.loads(task["prob"]))  # 各token的概率
        #print(task['label'])  # 1为泄漏，0为未泄漏
    ret1 = []
    ground_truth = []
    for task in dataset:
        ret1.append(min_k(epsilon=0.01, probs=json.loads(task["ground_truth_prob"])))
        #print(ret1)
        ground_truth.append(task['label'])
    #ret2 = latesteval.work()
    #ret3 = f1_llm.work()
    ret4 = f1_emb.work()

    #for r1, r2, r3, r4, task in zip(ret1, ret2, ret3, ret4, dataset):
        #acc1 += (task['label'] == r1)
        #acc2 += (task['label'] == r2)
        #acc3 += (task['label'] == r3)
        #acc4 += (task['label'] == r4)
    #for alpha in np.arange(0.01, 0.1, 0.01):
        #print("min-k%", evaluate_classification(ground_truth, [i>alpha for i in ret1], ret1))
    #print("latesteval", acc2 / len(dataset))
    #print("Rephrased Samples", acc3 / len(dataset))
    #print("embedding Samples", acc4 / len(dataset))
    for alpha in np.arange(0.1, 1, 0.1):
        print("embedding Samples", evaluate_classification(ground_truth, [i>alpha for i in ret4], ret4))
    print('original_gsm8k_data_contamination_detection_dataset_epoch2_only_llama')
    
    
    