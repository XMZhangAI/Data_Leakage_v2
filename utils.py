import json
def get_file_path(epoch=0, temp=0.0, output_dir='outputs_noset_1k'):
    return f'{output_dir}/HumanEval_samples_test_CodeLlama-7b-hf_f32_temp{temp}_test_epoch{epoch}.jsonl'

def read_jsonl_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            linedata = json.loads(line)
            if linedata['task_id'] not in data.keys():
                data[linedata['task_id']] = []
            data[linedata['task_id']].append(linedata['completion'])
    return data

# humaneval_data = get_dataset(dataset_name='humaneval')
def get_dataset(dataset_name='humaneval'):
    from datasets import load_from_disk
    dataset = load_from_disk(f'datasets/{dataset_name}')

    return dataset['test']

def get_copy_percentage(epoch=0, temp=0.0, dataset_name='humaneval', output_dir='outputs_noset_1k'):
    gendata =read_jsonl_file(get_file_path(epoch, temp, output_dir))
    humaneval_data = get_dataset(dataset_name)
    num=0
    totalnum=0
    for i in range(len(humaneval_data)):
        # print(humaneval_data[i]['canonical_solution'] in gready_data[i]['completion'])
        # print(humaneval_data[i]['canonical_solution'])
        # print(gready_data[i]['completion'])
        # print('------------------')
        # if  humaneval_data[i]['task_id'] in gendata.keys() and humaneval_data[i]['canonical_solution'].strip() in gendata[humaneval_data[i]['task_id']].strip():
        if  humaneval_data[i]['task_id'] in gendata.keys():
            for j in gendata[humaneval_data[i]['task_id']]:
                if humaneval_data[i]['canonical_solution'].strip() in j.strip():
                    num+=1
                    break
            totalnum+=1
    return num/totalnum

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

def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def get_edit_distance(epoch=0, temp=0.0, dataset_name='humaneval', model_path="/home/jiangxue/LLMs/CodeLlama-7b-hf", output_dir='outputs_noset_1k'):
    gendata =read_jsonl_file(get_file_path(epoch, temp, output_dir))
    humaneval_data = get_dataset(dataset_name)
    num=0
    totalnum=0
    tokenizer = load_tokenizer(model_path)
    for i in range(len(humaneval_data)):
        # print(humaneval_data[i]['canonical_solution'] in gready_data[i]['completion'])
        # print(humaneval_data[i]['canonical_solution'])
        # print(gready_data[i]['completion'])
        # print('------------------')
        # if  humaneval_data[i]['task_id'] in gendata.keys() and humaneval_data[i]['canonical_solution'].strip() in gendata[humaneval_data[i]['task_id']].strip():
        if  humaneval_data[i]['task_id'] in gendata.keys():
            for j in gendata[humaneval_data[i]['task_id']]:
                canonical_solution = tokenizer.encode(humaneval_data[i]['canonical_solution'].strip(), add_special_tokens=False)
                codesample = tokenizer.encode(j[len(humaneval_data[i]['prompt'].strip()):].strip(), add_special_tokens=False)[:len(canonical_solution)]
                # codesample = tokenizer.encode(j[len(humaneval_data[i]['prompt'].strip()):].strip(), add_special_tokens=False)
                num += levenshtein_distance(canonical_solution, codesample)
                # totalnum+=len(canonical_solution)
                # totalnum+= max(len(canonical_solution), len(codesample))
                totalnum+=1
    return num/totalnum

def get_edit_distance_distribution(epoch=0, temp=0.0, dataset_name='humaneval', model_path="/home/jiangxue/LLMs/CodeLlama-7b-hf", samplenum=50, output_dir = 'outputs_noset_1k', max_edit_distance = 1e9, addnum_key = True):
    gendata =read_jsonl_file(get_file_path(epoch, temp, output_dir))
    humaneval_data = get_dataset(dataset_name)
    num = []
    tokenizer = load_tokenizer(model_path)
    addnum = 0
    for i in range(len(humaneval_data)):
        if  humaneval_data[i]['task_id'] in gendata.keys():
            if addnum_key:
                addnum = samplenum - len(gendata[humaneval_data[i]['task_id']])
            minnum = 1e9
            for j in gendata[humaneval_data[i]['task_id']]:
                canonical_solution = tokenizer.encode(humaneval_data[i]['canonical_solution'].strip(), add_special_tokens=False)
                codesample = tokenizer.encode(j[len(humaneval_data[i]['prompt'].strip()):].strip(), add_special_tokens=False)[:len(canonical_solution)]
                if levenshtein_distance(canonical_solution, codesample)<=max_edit_distance:
                    edit_distance = levenshtein_distance(canonical_solution, codesample)
                    num.append(edit_distance)
                    minnum = min(minnum, edit_distance) 
            if minnum!=1e9:
                num.extend([minnum for _ in range(addnum)])
    return num

def get_ylim(n):
    import math
    ylength = math.ceil(max(n)*10) / 10
    return ylength

def plot_edit_distance(epochs=0, temp=0.0, dataset_name='humaneval', model_path="/home/jiangxue/LLMs/CodeLlama-7b-hf", samplenum=50, output_dir = 'outputs_noset_1k', max_edit_distance = 1e9):
    import matplotlib.pyplot as plt
    import numpy as np

    if type(epochs) == int:
        epochs = [epochs]

    # plt.figure(figsize=(10, 6))
    for epoch in epochs:
        num = get_edit_distance_distribution(epoch, temp, dataset_name, model_path, samplenum, output_dir, max_edit_distance)
        weights = np.ones_like(num) / len(num)
        n, bins = np.histogram(num, bins=50, weights=weights)
        plt.plot(bins[:-1], n, linestyle='-', marker='o', label=f'Epoch {epoch+1}')

    plt.ylim(0, get_ylim(n))
    plt.xlabel('Edit Distance')
    plt.ylabel('Frequency')
    plt.title(f'Edit Distance Distribution')
    plt.legend()
    plt.show()


def plot_pass_at_k(epochs=0, temp=0.8, dataset_name = 'humaneval', duplicates = True, output_dir = 'outputs_noset_1k', samplenum=50):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if type(epochs) == int:
        epochs = [epochs]

    # plt.figure(figsize=(10, 6))
    pass_at_1s = []
    copy_percentages = []
    fillkey = False
    for epoch in epochs:
        if not os.path.exists(get_file_path(epoch, temp, output_dir)):
            pass_at_1s.append(pass_at_k['pass@1'])
            copy_percentages.append(copy_percentage)
            fillkey = True
            continue
        pass_at_k = evaluate_pass_at_k(epoch, temp, dataset_name, duplicates, output_dir, samplenum)
        copy_percentage = get_copy_percentage(epoch, temp, dataset_name, output_dir)
        if fillkey:
            pass_at_1s[-1] = (pass_at_1s[-1]+pass_at_k['pass@1'])/2
            copy_percentages[-1] = (copy_percentages[-1]+copy_percentage)/2
            fillkey = False
        pass_at_1s.append(pass_at_k['pass@1'])
        copy_percentages.append(copy_percentage)
        print(epoch)
    
    plt.plot(range(len(pass_at_1s)), pass_at_1s, linestyle='-', marker='o', label=f'pass@1')
    plt.plot(range(len(copy_percentages)), copy_percentages, linestyle='-', marker='o', label=f'copy_percentage')
    

    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Data Leakage')
    plt.legend()
    plt.show()

def get_edit_distance_distribution_dict(epoch=0, temp=0.0, dataset_name='humaneval', model_path="/home/jiangxue/LLMs/CodeLlama-7b-hf", samplenum=50, output_dir = 'outputs_noset_1k', max_edit_distance = 1e9, addnum_key = True):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gendata =read_jsonl_file(get_file_path(epoch, temp, output_dir))
    humaneval_data = get_dataset(dataset_name)
    tokenizer = load_tokenizer(model_path)
    num = {}
    min_lengths = {}
    addnum = 0
    for i in range(len(humaneval_data)):
        task_id = humaneval_data[i]['task_id']
        if  task_id in gendata.keys():
            num[task_id] = []
            min_lengths[task_id] = []
            if addnum_key:
                addnum = samplenum - len(gendata[task_id])
            minnum = 1e9
            canonical_solution = tokenizer.encode(humaneval_data[i]['canonical_solution'].strip(), add_special_tokens=False)
            for j in gendata[task_id]:
                codesample = tokenizer.encode(j[len(humaneval_data[i]['prompt'].strip()):].strip(), add_special_tokens=False)
                min_length = len(canonical_solution) #min(len(canonical_solution), len(codesample))
                canonical_solution, codesample = canonical_solution[:min_length], codesample[:min_length]
                if levenshtein_distance(canonical_solution, codesample)<=max_edit_distance:
                    edit_distance = levenshtein_distance(canonical_solution, codesample)
                    num[task_id].append(edit_distance)
                    min_lengths[task_id].append(min_length)
                    minnum = min(minnum, edit_distance) 
            if minnum!=1e9:
                num[task_id].extend([minnum for _ in range(addnum)])
    return num, min_lengths

def evaluate_pass_at_k(epoch=0, temp=0.0, dataset_name = 'humaneval', duplicates = True, top_percent_exclusion = 0, output_dir = 'outputs_noset_1k', samplenum = 50, punish = False):
    from evaluate.execute.execution import evaluate_with_test_code
    from evaluate.evaluation import pass_at_K, AvgPassRatio
    import numpy as np
    humaneval_data = get_dataset(dataset_name)
    INPUT_PATH = get_file_path(epoch, temp, output_dir)

    with open(INPUT_PATH, 'r') as f:
        data_dict = {}
        handled_solutions = []
        edges = {}
        task_index = {}
        for idx, task in enumerate(humaneval_data):
            data_dict[task['task_id']] = task
            task_index[task['task_id']] = 0
        if top_percent_exclusion>0:
            bin_num = int(100/top_percent_exclusion)
            edit_distance_distribution_dict, min_lengths = get_edit_distance_distribution_dict(epoch, temp, dataset_name, samplenum=samplenum, output_dir = output_dir, addnum_key=False)
            for key in edit_distance_distribution_dict.keys():
                edit_distance_distribution = edit_distance_distribution_dict[key]
                weights = np.ones_like(edit_distance_distribution) / len(edit_distance_distribution)
                n, bin_edges = np.histogram(edit_distance_distribution, bins=bin_num, weights=weights)
                index = np.argmax(np.cumsum(n) > float(top_percent_exclusion)/100)
                edge = bin_edges[index+1]
                edges[key] = edge
            print(edges)
        for line in f:
            line = json.loads(line)
            task_id = line["task_id"]
            line["prompt"] = ""
            line["test"] = data_dict[task_id]["test"]
            line["entry_point"] = data_dict[task_id]["entry_point"]
            task_index[task_id] += 1
            if not duplicates and line in handled_solutions:
                continue
            if top_percent_exclusion>0 and edit_distance_distribution_dict[task_id][task_index[task_id]-1]<= \
                min(edges[task_id], int(min_lengths[task_id][task_index[task_id]-1]*float(top_percent_exclusion)/100)+1):
                continue
            handled_solutions.append(line)
    exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
        
    result = pass_at_K(exec_result, k=[1,5,10])
    if punish:
        handled_task = list(set([i["task_id"] for i in handled_solutions]))
        percentage = len(handled_task) / len(data_dict.keys())
        for key in result.keys():
            result[key] = result[key] * percentage
    
    return result

def get_pass_at_k_list(epochs=0, temp=0.8, dataset_name = 'humaneval', duplicates = True, output_dir = 'outputs_noset_1k', samplenum = 50, top_percent_exclusion = 0, punish = False):
    import os
    pass_at_1s = []
    fillkey = False
    for epoch in epochs:
        if not os.path.exists(get_file_path(epoch, temp, output_dir)):
            pass_at_1s.append(pass_at_k['pass@1'])
            fillkey = True
            continue
        pass_at_k = evaluate_pass_at_k(epoch, temp, dataset_name, duplicates, top_percent_exclusion, output_dir, samplenum, punish)
        if fillkey:
            pass_at_1s[-1] = (pass_at_1s[-1]+pass_at_k['pass@1'])/2
            fillkey = False
        pass_at_1s.append(pass_at_k['pass@1'])
        print(pass_at_1s[-1])
    return pass_at_1s

def build_test_method_for_apps(input_output, test_case_limit = 5):
    test = "def check(candidate):\n"
    for idx, (input, output) in enumerate(zip(input_output['inputs'], input_output['outputs'])):
        if idx >= test_case_limit:
            break
        try:
            test += "\tassert candidate(%r) == %r \n" % (input.strip(), output.strip())
        except:
            test += "\tassert candidate(%s) == %s \n" % (input, output)
    return test

import ast
def find_method_name(code, lang="python"):
    if lang == "python":
        try:
            # method_name = code.split("def ")[-1].split("(")[0]
            parsed = ast.parse(code)
            function_defs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
            if function_defs:
                if len(function_defs) == 1:
                    method_name = function_defs[0].name
                else:
                    method_name = function_defs[-1].name if function_defs[-1].name != "main" else function_defs[-2].name
            else:
                method_name = None
        except:
            method_name = None

    return method_name