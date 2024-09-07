import re
import openai
import time
import copy
import json
import argparse
import tqdm
import random
random.seed(42)

import sys
sys.path.append('/home/jiangxue/few-shot-prompting-for-code')

from pal import interface
from pal.prompt import mbpp_prompts_10
from datasets import load_dataset

from utils import code_split, build_test_method_for_apps, insert_test_case
from execute.execution import evaluate_with_test_code
from evaluation import pass_at_K


OUTPUT_PATH = 'eval_results/0209_apps_eval.jsonl'
from datasets import load_dataset
test_dataset = load_dataset("codeparrot/apps", split="test", difficulties=["introductory"])


parser = argparse.ArgumentParser()
parser.add_argument('--fail_list', type=list, default=[])
parser.add_argument('--append', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument("--timeout", type=float, default=10, help="how many seconds to wait during execution for each test case")
parser.add_argument("--public_case_num", type=int, default=9)
args = parser.parse_args()

itf = interface.ProgramInterface(
    stop='\n\n',
    verbose=args.verbose
)

handled_solutions = []
fail_list = []


with open(OUTPUT_PATH, 'a' if args.append else 'w') as f:
    pbar = tqdm.tqdm(test_dataset, total=len(test_dataset))
    for problem in pbar:

        if args.append and (problem['problem_id'] not in args.fail_list):
            continue

        question = problem['question']
        m = re.search("\-+Example", question)
        # if problem['solutions']:
        #     solutions = json.loads(problem['solutions'])
        # else:
        #     solutions = []
        input_output = json.loads(problem['input_output'])
        entry_point = 'solution'
        test = build_test_method_for_apps(input_output, test_case_limit = 5)
        
        if m is None:
            prompt = "'''" + question + "'''" + "\ndef solution(stdin: str) -> str:\n"
        else:
            prompt = "'''" + question[:m.start()] + "'''" + "\ndef solution(stdin: str) -> str:\n"

        try:
            # code_snippets = itf.run(mbpp_prompts_10.MBPP_PROMPT.format(x_test=prompt), majority_at = 1, max_tokens=1024)
            code_snippets = itf.run(prompt, majority_at=1, max_tokens=1024)
        except RuntimeError as e:
            print(str(e))
            print("problem-%d fail"%(problem['task_id']))
            fail_list.append(problem['task_id'])
            continue
        except openai.error.ServiceUnavailableError as e:
            print(str(e))
            print("problem-%d fail"%(problem['task_id']))
            fail_list.append(problem['task_id'])
            time.sleep(120)
            continue

        for code in code_snippets:
            solution = {
                'task_id': problem['problem_id'],
                'prompt': prompt,
                'test': test,
                'entry_point': entry_point,
                'completion': code
            }
            handled_solutions.append(solution)
            exec_result = evaluate_with_test_code([solution], timeout=args.timeout)

        f.write(json.dumps(exec_result[0]) + '\n')
        
        itf.clear_history()
        f.flush()


exec_result = evaluate_with_test_code(handled_solutions, timeout=args.timeout)
with open(OUTPUT_PATH, 'a' if args.append else 'w') as f:
    for result in exec_result:
        f.write(json.dumps(result) + '\n')
    f.flush()

print('pass rates of solutions')
pass_at_K(exec_result, k=[1,2])
