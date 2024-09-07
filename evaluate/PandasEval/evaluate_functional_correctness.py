import sys

from evaluation import evaluate_functional_correctness

sample_file = "/home/jiangxue/Fast_Training_for_Code_Generation/select/outputs/MonkeyEval_direct_output_Llama-2-7b-hf_temp0.0_train.jsonl"
problem_file = "/home/jiangxue/datasets/MonkeyEval/real_monkey_eval_v3.jsonl_train"
k = [1]
timeout = 10
n_workers = 1
results = evaluate_functional_correctness(sample_file, problem_file, k, n_workers, timeout)
print(results)
