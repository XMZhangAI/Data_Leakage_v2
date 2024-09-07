import json
import os
from datasets import load_from_disk
from utils import find_method_name

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))   
    return data

def truncate_ori(d):
    pred = d[:d.find('def')]
    d = d[d.find('def'):].split('\n\n')
    s = pred + d[0] + '\n\n'
    if len(d)>1:
        for i in d[1:]:
            if 'def' not in i and i and '__main__' not in i:
                s += i + '\n\n'
            else:
                break
    return s
# (not i.startswith('def')) 

import re
# def truncate(d, method_name):
#     pred = d[:d.find('def '+method_name)]
#     d = d[d.find('def '+method_name):].split('\n\n')
#     s = pred + d[0] + '\n\n'
#     if len(d)>1:
#         for i in d[1:]:
#             if (('def' not in i) or ("  def" in i)) and i and '__main__' not in i:
#                 s += i + '\n\n'
#             else:
#                 break
#     return s

def truncate(d, method_name):
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

if __name__ == '__main__':
    path = '/home/zhangxuanming/DataLeakage_v2/outputs_codegen-6B_new_humaneval_10k_0217'
    write_path = os.path.join(path,'truncated')
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    data_dict = {}
    dataset = load_from_disk("datasets/humaneval")
    for idx, task in enumerate(dataset["test"]):
        data_dict[task['task_id']] = task    
    for file in os.listdir(path):
        #if file not dictionary:
        if not file.endswith('.jsonl'):
            continue
        filepath = os.path.join(path, file)
        data = read_jsonl_file(filepath)
            
        for i in range(len(data)):
            data[i]['completion'] =  truncate(data[i]['completion'], data_dict[data[i]["task_id"]]["entry_point"])

        recordpath = os.path.join(write_path, file)
        with open(recordpath, 'w') as file:
            for record in data:
                # 将字典转换为 JSON 字符串并写入文件
                json_record = json.dumps(record)
                file.write(json_record + '\n')
    
    # path = 'outputs/HumanEval_samples_test_CodeLlama-7b-hf_f32_temp0.0_test_epoch-1.jsonl'

    # data = read_jsonl_file(path)
        
    # for i in range(len(data)):
    #     data[i]['completion'] =  truncate(data[i]['completion'])

    # recordpath = path[:-6] + '_truncated.jsonl'
    # with open(recordpath, 'w') as file:
    #     for record in data:
    #         # 将字典转换为 JSON 字符串并写入文件
    #         json_record = json.dumps(record)
    #         file.write(json_record + '\n')