# import sys
# sys.path.append('/home/jiangxue/Fast_Training_for_Code_Generation')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import argparse
import json
import logging
import openai
import pprint
import re
import time
import torch
# from jaxformer.hf import sample
# from jaxformer.hf.codegen import modeling_codegen
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, CodeLlamaTokenizer
from peft import PeftModel

def format_prompt(task_id, text, tests, sample_code, num_prompts):
    # Create prompt from scratch
    prompt = f'"""\n{text}\n\n'
    if num_prompts > 0:
        for i in range(num_prompts):
            example = tests[i].split("assert ")[-1].replace("==", "=")
            prompt += f">>> Example: {example}\n"

    # Add code prefix
    fn_name = tests[0].split("assert ")[-1].split("(")[0]
    fn_name = fn_name.strip()
    fn_search = re.search(f"def {fn_name}\s?\(.*\)\s?:", sample_code)
    
    if fn_search is None:
        raise ValueError(
            f"Could not find 'def {fn_name}\(.*\):' in code for task {task_id}."
        )
    code_prefix = sample_code[: fn_search.end()]
    prompt = f'{prompt}"""\n\n{code_prefix}\n'
    return prompt


def sample_code_from_llm(args, prompt, completion, model, tokenizer, device, return_probs=False):
    completions = []
    completions_probs = []
    prompt_token_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens= False, verbose=False) 
    source_input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt+completion, add_special_tokens= False, verbose=False)
    target_input_ids = source_input_ids[len(prompt_token_ids):]

    model.eval()

    try:
        with torch.no_grad():
            for id, input_id in enumerate(target_input_ids):
                input_id = torch.tensor(prompt_token_ids + target_input_ids[:id]).unsqueeze(0).to(device[0])
                tokens = model.generate(
                    input_id,
                    num_return_sequences=1,
                    max_length=input_id.shape[1]+1,
                    use_cache=True,
                    do_sample=False,
                    return_dict_in_generate=True, 
                    output_scores=True
                )
                scores = tokens.scores[0]
                probs = torch.softmax(scores, dim=-1)
                generated_token_prob = probs[0, target_input_ids[id]].item()
                completions_probs.append({tokenizer.decode(target_input_ids[id]):generated_token_prob})
            

    except RuntimeError as e:
        logging.error(f"Could not sample from model: {e}")
    
    generated_text = completion
    completions=[(generated_text, completions_probs)]
    return completions #list(set(completions))

def initialize_openai(args):
    api_key = open(f"{args.openai_creds_dir}/openai_api_key.txt").read()
    openai.organization = open(
        f"{args.openai_creds_dir}/openai_organization_id.txt"
    ).read()
    openai.api_key = api_key


def write_jsonl(data, output_filepath):
    with open(output_filepath, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

def load_model_tokenizer(args, arch, model_path, model_dir, device):
    if "codegen" in arch:
        base_model = AutoModelForCausalLM.from_pretrained(
            f"Salesforce/{args.arch}", low_cpu_mem_usage=True, torch_dtype=torch.float32,
            device_map={"": args.gpu[0]}
        )
        if args.model_path:
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
        else:
            model = base_model 

        # model = None        

        tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/{args.arch}", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        generate_code_fn = lambda args, prompt, completion: sample_code_from_llm(
            args, prompt, completion, model, tokenizer, device, args.return_probs
        )

    elif "Llama" in arch:
        base_model = AutoModelForCausalLM.from_pretrained(
            f"{model_dir}",
            low_cpu_mem_usage=True,
            revision="float32",
            torch_dtype=torch.float32,
            device_map={"": args.gpu[0]}
        )

        # base_model = None

        # base_model = LlamaForCausalLM.from_pretrained(f"{model_dir}/{arch}",
        #                                                 revision="float16",
        #                                                 torch_dtype=torch.float16,
        #                                                 low_cpu_mem_usage=True,
        #                                             ).cuda()
        if args.model_path:
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
        else:
            model = base_model

        tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        generate_code_fn = lambda args, prompt, completion: sample_code_from_llm(
            args, prompt, completion, model, tokenizer, device, args.return_probs
        )
    
    elif "Mistral" in arch:
        base_model = AutoModelForCausalLM.from_pretrained(
            f"/home/dongyh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
            low_cpu_mem_usage=True,
            revision="float32",
            torch_dtype=torch.float32,
            device_map={"": args.gpu[0]}
        )

        if args.model_path:
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
        else:
            model = base_model

        tokenizer = AutoTokenizer.from_pretrained(f"/home/dongyh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        generate_code_fn = lambda args, prompt, completion: sample_code_from_llm(
            args, prompt, completion, model, tokenizer, device, args.return_probs
        )

    return generate_code_fn

def generate_code_for_problems(args, except_tasks, output_filepath):
    if args.dataset == "MBPP":
        dataset = load_from_disk("datasets/mbpp_sanitized") #load_dataset("mbpp")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif args.dataset == "HumanEval":
        dataset = load_from_disk('datasets/humaneval')
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif args.dataset == "GSM8K":
        dataset = load_from_disk('datasets/test_gsm8k_198')
    elif args.dataset == "NewHumanEval":
        dataset = load_from_disk('datasets/new_humaneval')
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])

    # output = []
    if torch.cuda.is_available():  
        device = [f"cuda:{gpu}" for gpu in args.gpu]
    else:  
        device = torch.device("cpu")

    generate_code_fn = load_model_tokenizer(args, args.arch, args.model_path, args.model_dir, device)
    f = open(output_filepath, "a")

    for i in tqdm(range(len(dataset))):
        if (dataset["task_id"][i] in except_tasks):
            continue
        try:
            if args.dataset == "MBPP":
                prompt = format_prompt(
                    dataset["task_id"][i],
                    dataset["prompt"][i],
                    dataset["test_list"][i],
                    dataset["code"][i],
                    args.num_use_cases,
                )
                completion = dataset["code"][i]
            else:
                # humaneval
                prompt = dataset["prompt"][i]
                completion = dataset["canonical_solution"][i]
        except ValueError as e:
            logging.error(e)
            continue

        
        task_id = dataset["task_id"][i]
        for completion in generate_code_fn(args, prompt, completion):
            output = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": completion[0],
                    "prob": completion[1]
                }
            f.write(json.dumps(output) + "\n")
            f.flush()            

    f.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained model to generate Python code for the MBPP benchmark."
    )
    parser.add_argument(
        "--arch",
        default="gptj",
    )
    parser.add_argument(
        "--model-dir",
        default="checkpoints",
        help="Directory where pre-trained CodeGen model checkpoints are saved.",
    )
    parser.add_argument(
        "--model-path",
        help="Directory to load model checkpoint from. If None, will load a pre-trained "
        "CodeGen model using the --arch argument instead.",
        default=None,
    )
    parser.add_argument("--acctual-num-samples", default=10, type=int)
    parser.add_argument('--gpu', nargs='+', type=int, default=[0], help='GPU devices to use')
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--return_probs", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="Which MBPP split to use. In datasets v1.16.1, MBPP only has the split 'test'.",
    )
    parser.add_argument(
        "-s", "--start", default=1, type=int, help="Task ID to start with."
    )
    parser.add_argument(
        "-e", "--end", default=975, type=int, help="Task ID to end with (exclusive)."
    )
    parser.add_argument(
        "-n",
        "--num-use-cases",
        default=1,
        type=int,
        help="Number of assert (test examples) to give in the task description.",
    )
    parser.add_argument(
        "--max-request-time",
        type=int,
        default=80,
        help="Max. time to wait for a successful GPT-3 request.",
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=10,
        help="Time to sleep (in seconds) between each GPT-3 call.",
    )
    parser.add_argument(
        "--openai-creds-dir",
        type=str,
        default=None,
        help="Directory where OpenAI API credentials are stored. Assumes the presence of "
        "openai_api_key.txt and openai_organization_id.txt files.",
    )
    parser.add_argument("--dataset", default="MBPP", type=str)
    args = parser.parse_args()
    return args

def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.model_path:
        output_filepath = os.path.join(
            args.output_dir,
            f"{args.dataset}_samples_{args.split}_{args.arch}_f32_temp{args.temperature}_{args.output_file_suffix}_{args.model_path.split('-')[-1]}.jsonl",
        )   
    else:
        output_filepath = os.path.join(
            args.output_dir,
            f"{args.dataset}_samples_{args.split}_{args.arch}_f32_temp{args.temperature}_{args.output_file_suffix}_epoch-1.jsonl",
        )  
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    except_tasks = []
    if os.path.exists(output_filepath):
        print(f"File {output_filepath} already exists in {args.output_dir}.")
        lines = open(output_filepath).readlines()
        for line in lines:
            task_id = json.loads(line)["task_id"]
            if task_id not in except_tasks:
                except_tasks.append(task_id)

    completions = generate_code_for_problems(args, except_tasks, output_filepath)

    # write_jsonl(completions, output_filepath)

if __name__ == "__main__":
    main(parse_args())


