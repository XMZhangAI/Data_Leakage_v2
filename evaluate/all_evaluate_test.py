import re
import json
import copy
import argparse
import sys
from post_process import post_process_code
sys.path.append("/home/dongyh/A100/self-collaboration-code-gen")

from utils import build_test_method, build_AVG_solutions, find_method_name, code_split, build_test_method_for_apps
from execute.execution import evaluate_with_test_code, evaluate_with_test_code_T
from evaluation import pass_at_K, AvgPassRatio
from datasets import load_dataset, load_from_disk
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mbpp')
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--input_path', type=str, default='result/gpt-3.5-turbo-1106_0.0_1_codeforces.json')
parser.add_argument('--output_path', type=str, default='test')
parser.add_argument('--duplicates', type=bool, default=True)
args = parser.parse_args()

INPUT_PATH = args.input_path
OUTPUT_PATH = args.output_path

if args.dataset == 'mbpp-sanitized':
    dataset = load_from_disk("datasets/mbpp_sanitized")
    # dataset = load_dataset("mbpp", "sanitized")
    dataset_key = ["train","test","validation","prompt"]
elif args.dataset == 'mbpp':
    dataset = load_from_disk("/home/jiangxue/datasets/MBPP-full")
    # dataset = load_from_disk("datasets/mbpp_full")
    dataset_key = ["train","test","validation","prompt"]
elif args.dataset == 'humaneval':
    dataset = load_from_disk("datasets/humaneval")
    # dataset = load_from_disk("/mnt/c/Users/17385/Desktop/my_datasets/humaneval")
    # dataset = load_dataset("openai_humaneval")
    dataset_key = ["test"]
elif args.dataset == 'humanevalx':
    dataset = load_dataset("THUDM/humaneval-x", args.lang)
    dataset_key = ["test"]
elif args.dataset == 'codeforces':
    dataset = load_from_disk("datasets/codeforces")
    dataset_key = ["test"]

with open(INPUT_PATH, 'r') as f:
    if args.dataset == 'humaneval':
        except_list = []#["HumanEval/39", "HumanEval/137", "HumanEval/141", "HumanEval/33"]
    if args.dataset == 'mbpp-sanitized':
        except_list = []

        # # -----------------------------------
        # handled_solutions = []
        # for idx, line in enumerate(f):
        #     line = json.loads(line)
        #     line["task_id"] = int(line["task_id"])
        #     if line["task_id"] in except_list:
        #         continue
        #     handled_solutions.append(line)
        # # -----------------------------------

    # data_dict = {}
    handled_solutions = []
    for key in dataset_key:
        for idx, task in enumerate(dataset[key]):
            task["prompt"] = ""
            task["completion"] = task["code"]
            task["entry_point"] = find_method_name(task["code"])
            task["test"] = build_test_method(task['test_list'], task['test_setup_code'], task['entry_point'])
            handled_solutions.append(task)
            

    # for line in f:
    #     line = json.loads(line)
    #     if args.dataset == 'codeforces':
    #         line["test"] =  build_test_method_for_apps(line["test"], test_case_limit = 5)
    #         line["entry_point"] = 'solution'
    #     else:
    #         line["test"] = data_dict[line["task_id"]]["test"]
    #         line["entry_point"] = data_dict[line["task_id"]]["entry_point"]
    #     # ChatGPT
    #     line["completion"] = post_process_code(prompt=line['prompt'], code=line['completion'], func_name=line['entry_point'], m_indent='    ')
    #     # CodeLlame
    #     # line["completion"] = line['completion']
        
    #     line["prompt"] = ""
    #     if not args.duplicates and line in handled_solutions:
    #         continue
    #     handled_solutions.append(line)


# data_dict = {}
# for key in dataset_key:
#     for idx, task in enumerate(dataset[key]):
#         data_dict[task['task_id']] = task

# # ---------------------------------------------------
# # repare for baseline evaluation
# for idx, solution in enumerate(handled_solutions):
#     raw_data = data_dict[solution["task_id"]]
#     raw_data["completion"] = solution["completion"]
#     handled_solutions[idx] = raw_data
#     if args.dataset == 'mbpp':
#         handled_solutions[idx]["prompt"] = ""
#         handled_solutions[idx]["entry_point"] = find_method_name(solution["completion"])
#         method_name, signature, _ , func_body, before_func = code_split(raw_data['code'])
#         handled_solutions[idx]["test"] = build_test_method(raw_data['test_list'], raw_data['test_setup_code'], method_name)

# ---------------------------------------------------
# # CodeBLEU & BLEU
# pre_list = []
# ref_list = []
# for idx, solution in enumerate(handled_solutions):
#     if (args.dataset == 'mbpp') or (args.dataset == 'mbpp-sanitized'):    
#         pre = "def" + solution["completion"]
#         ref = data_dict[solution["task_id"]]["code"]
#     elif args.dataset == "humaneval":
#         pre = solution["completion"]
#         ref = data_dict[solution["task_id"]]["canonical_solution"]
#     elif args.dataset == "humanevalx":
#         pre = solution["generation"]
#         ref = data_dict[solution["task_id"]]["canonical_solution"]
    
#     # pre = re.sub("\n+", "\n", pre)
#     # ref = re.sub("\n+", "\n", ref)

#     pre_list.append(pre)
#     ref_list.append([ref])

#     # # Levenshtein Distance
#     # handled_solutions[idx]["levenshtein"] = Levenshtein_Distance(pre, ref)

# # bleu = evaluate.load("bleu")
# codebleu = evaluate.load("dvitel/codebleu")
# # bleu_score = bleu.compute(predictions = pre_list, references = ref_list)
# codebleu_score = codebleu.compute(predictions = pre_list, references = ref_list, lang = args.lang)
# # print("BLEU: ", bleu_score["bleu"])
# print("CodeBLEU: ", codebleu_score["CodeBLEU"])
# ---------------------------------------------------

# ---------------------------------------------------
# AvgPassRatio
# handled_solutions_AVG = build_AVG_solutions(handled_solutions)
# exec_result_AVG = evaluate_with_test_code(handled_solutions_AVG, timeout=10)
# avg_pass_ratio = AvgPassRatio(exec_result_AVG)
# print("AvgPassRatio: ", avg_pass_ratio)
# ---------------------------------------------------

# ---------------------------------------------------
# sum_levenshtein_distance = 0
# length = len(handled_solutions)
# for idx, solution in enumerate(handled_solutions):
#     sum_levenshtein_distance += solution["levenshtein"]
# print("Average Levenshtein Distance: ", sum_levenshtein_distance / length)
# ---------------------------------------------------

# for solution in handled_solutions:
#     solution["entry_point"] = find_method_name(solution["completion"]) if find_method_name(solution["completion"]) else "candidate"

# ---------------------------------------------------
# pass@1
exec_result = evaluate_with_test_code(handled_solutions, timeout=1)
# with open(INPUT_PATH+"_results.jsonl", 'w') as f:
#     for idx, result in enumerate(exec_result):
#         f.write(json.dumps(result) + '\n')
#     f.flush()
# with open(INPUT_PATH+"_results_summary.jsonl", 'w') as f:
#     f.write(json.dumps(pass_at_K(exec_result, k=[1,5,10])) + '\n')
# print("CodeBLEU: ", codebleu_score["CodeBLEU"])
# print("AvgPassRatio: ", avg_pass_ratio)
print('pass rates of solutions')
print(len(exec_result))
print(pass_at_K(exec_result, k=[1]))
# # ---------------------------------------------------


# # ---------------------------------------------------
# pass@k
# GREEDYPATH = 'eval_results/formal_result/0308_humaneval_single_turn_with_ground_truth_plan_gen.jsonl'
# with open(GREEDYPATH, 'r') as f:
#     handled_solutions_greedy = [json.loads(line) for line in f]

# # group by task_id
# handled_solutions_dict = {}
# for solution in handled_solutions_greedy:
#     if solution["task_id"] not in handled_solutions_dict:
#         handled_solutions_dict[solution["task_id"]] = []
#     handled_solutions_dict[solution["task_id"]].append(solution)

# for solution in handled_solutions:
#     handled_solutions_dict[solution["task_id"]].append(solution)

# # pass@2
# handled_solutions_2 = []
# for key in handled_solutions_dict:
#     handled_solutions_2 += handled_solutions_dict[key][:2]
# exec_result_2 = evaluate_with_test_code(handled_solutions_2, timeout=10)

# # pass@5
# handled_solutions_5 = []
# for key in handled_solutions_dict:
#     handled_solutions_5 += handled_solutions_dict[key][:5]
# exec_result_5 = evaluate_with_test_code(handled_solutions_5, timeout=10)

# # pass@10
# handled_solutions_10 = []
# for key in handled_solutions_dict:
#     handled_solutions_10 += handled_solutions_dict[key][:10]
# exec_result_10 = evaluate_with_test_code(handled_solutions_10, timeout=10)

# print('pass@2')
# pass_at_K(exec_result_2, k=[2])
# print('pass@5')
# pass_at_K(exec_result_5, k=[5])
# print('pass@10')
# pass_at_K(exec_result_10, k=[10])
# # ---------------------------------------------------



# exec_result_s = copy.deepcopy(exec_result)

# handled_solutions_APR = copy.deepcopy(handled_solutions)



# ---------------------------------------------------
# # More Test Cases
# if (args.dataset == 'mbpp') or (args.dataset == 'mbpp-sanitized'):
#     test_case_path= 'data/mbpp_full_test_case.json'
#     with open(test_case_path, 'r') as f:
#         test_cases = json.load(f)
    
#     test_cases_dict = {}
#     for case in test_cases:
#         test = build_test_method(case['test_list'], case['test_setup_code'], case['entry_point'])
#         test_cases_dict[case['task_id']] = test

# elif args.dataset == "humaneval":
#     test_case_path= 'data/HumanEval_test_case.jsonl'
#     with open(test_case_path, 'r') as f:
#         test_cases = [json.loads(line) for line in f]
        
#     test_cases_dict = {}
#     for case in test_cases:
#         test = build_test_method(case['test_case_list'], "", case['entry_point'])
#         test_cases_dict[case['task_id']] = test


# for solution in handled_solutions:
#     solution['test'] =test_cases_dict[solution['task_id']]


# # # AvgPassRatio
# # handled_solutions_AVG = build_AVG_solutions(handled_solutions)
# # exec_result_AVG = evaluate_with_test_code(handled_solutions_AVG, timeout=10)
# # avg_pass_ratio = AvgPassRatio(exec_result_AVG)
# # print("AvgPassRatio: ", avg_pass_ratio)

# # ---------------------------------------------------

# exec_result_T = evaluate_with_test_code(handled_solutions, timeout=10)

# # with open(OUTPUT_PATH, 'w') as f:
# #     for idx, result in enumerate(exec_result_s):
# #         result["result_T"] = exec_result_T[idx]["result"]
# #         result["passed_T"] = exec_result_T[idx]["passed"]
# #         f.write(json.dumps(result) + '\n')
# #     f.flush()

# # print("AvgPassRatio: ", avg_pass_ratio)
# # print('pass rates of solutions')
# # pass_at_K(exec_result, k=[1])

# print('pass rates - T')
# pass_at_K(exec_result_T, k=[1])