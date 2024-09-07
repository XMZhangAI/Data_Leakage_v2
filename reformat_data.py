from datasets import load_from_disk, concatenate_datasets

def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def filter_dataset(dataset, tokenizer, samplenum=10000, max_seq_length=1024, reverse=False):
    index = []
    num = 0
    tokenNum = 0
    for i in range(len(dataset)):
        if len(dataset[i]['canonical_solution'])>20000:
            continue
        input_token_ids = tokenizer.encode(dataset[i]['canonical_solution'], verbose=False)
        if len(input_token_ids)<=max_seq_length: #and len(input_token_ids) > max_seq_length/4*3:
            index.append(i)
            num += 1
            tokenNum +=len(input_token_ids)
            if num == samplenum:
                break
    print(tokenNum)
    return dataset.select(index)

def add_new_column(example):
    if "prompt" not in example.keys():
        example["prompt"] = ""
    if "test" not in example.keys():
        example["test"] = ""
    if "entry_point" not in example.keys():
        example["entry_point"] = ""
    return example

def add_index(examples, idx):
    # 添加序号列
    examples["task_id"] = idx
    return examples


def union_humaneval_starcoder_data(dataset_path='datasets/starcoder_data_100K', samplenum=10000):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.remove_columns(["max_stars_repo_path", "max_stars_repo_name", 'max_stars_count'])
    dataset = dataset.rename_column('id', 'task_id')
    dataset = dataset.rename_column('content', 'canonical_solution')       
    tokenizer = load_tokenizer(model_path="/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6")
    dataset = filter_dataset(dataset, tokenizer, samplenum)   
    dataset = dataset.map(add_new_column)
    
    # 需要更换路径
    leakage_dataset = load_from_disk('/home/zhangxuanming/DataLeakage_v2/dataset_gsm')
    
    dataset = concatenate_datasets([dataset, leakage_dataset['test']])
    dataset.shuffle(seed=42)
    
    save_path = dataset_path + '_new_humaneval_reformat_test_' + str(samplenum)
    dataset.save_to_disk(save_path)
    
    return dataset

def union_gsm8k_red_pajama_data(dataset_path='/home/zhangxuanming/DataLeakage_v2/datasets/red_pajama_data_100K', samplenum=10000):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.remove_columns(["meta"])
    dataset = dataset.rename_column('text', 'canonical_solution')
    tokenizer = load_tokenizer(model_path="/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6")
    dataset = filter_dataset(dataset, tokenizer, samplenum, reverse=True)
    dataset = dataset.map(add_new_column)
    dataset = dataset.map(add_index, with_indices=True)
    
    leakage_dataset = load_from_disk('/home/zhangxuanming/DataLeakage_v2/datasets/gsm8k_main')
    leakage_dataset = leakage_dataset["test"]
    leakage_dataset = leakage_dataset.rename_column('question', 'prompt')
    leakage_dataset = leakage_dataset.rename_column('answer', 'canonical_solution')
    leakage_dataset = filter_dataset(leakage_dataset, tokenizer, 198)
    leakage_dataset = leakage_dataset.map(add_new_column)
    leakage_dataset = leakage_dataset.map(add_index, with_indices=True)
    
    dataset = concatenate_datasets([dataset, leakage_dataset])
    dataset.shuffle(seed=42)
    
    save_path = dataset_path + '_gsm8k_reformat_' + str(samplenum)
    dataset.save_to_disk(save_path)
    
    return dataset

def reformat_dataset(dataset_leakage, dataset, samplenum=10000):
    if dataset_leakage == "gsm8k":
        return union_gsm8k_red_pajama_data(samplenum=samplenum)
    else:
        return union_humaneval_starcoder_data(samplenum=samplenum)

if __name__ == '__main__':
    #reformat_dataset(dataset_leakage="humaneval", dataset="starcoder_data", samplenum = 10000)
    reformat_dataset(dataset_leakage="gsm8k", dataset="red_pajama_data", samplenum = 10000)