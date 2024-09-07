'''
相比于原始版本，控制所有tokenizer增加bos token

'''

# import sys
# sys.path.append('/home/jiangxue/Fast_Training_for_Code_Generation')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

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
    code_prefix = sample_code[:fn_search.end()]
    prompt = f'{prompt}"""\n\n{code_prefix}\n'
    return prompt


def sample_code_from_llm(args, prompt, model, tokenizer, device, return_probs=False):
    completions = []

    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(str(prompt), add_special_tokens=False, verbose=False) 
    input_ids = torch.tensor([input_ids]).to(device[0])
    

    # FIXME: 
    num_return_sequences = args.acctual_num_samples

    if args.temperature == 0.0:
        args.num_samples = 1
        num_return_sequences = 1

    model.eval()

    for i in range(int(args.num_samples/num_return_sequences)):
        try:
            # Note: max_length is max length of input IDs, and max_length_sample is max length for completion (not including input IDs)
            if args.temperature > 0:
                tokens = model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    max_length=input_ids.shape[1] + 300,
                    temperature=args.temperature,
                    use_cache=True,
                )
            else:
                if return_probs:
                    tokens = model.generate(
                        input_ids,
                        num_return_sequences=1,
                        max_length=input_ids.shape[1] + 300,
                        use_cache=True,
                        do_sample=False,
                        return_dict_in_generate=True, 
                        output_scores=True
                    )
                    generated_ids = tokens.sequences[0]
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    completions_probs = []
                    # TODO: score未除以temperature
                    for step, scores in enumerate(tokens.scores):
                        # 在每一步，scores是对下一个token的logits
                        # 计算softmax以得到概率分布
                        probs = torch.softmax(scores, dim=-1)
                        # 获取生成的token的索引
                        generated_token_id = generated_ids[step + input_ids.shape[1]] # 偏移1因为生成的第一个token是基于输入文本的
                        # 获取实际生成token的概率
                        generated_token_prob = probs[0, generated_token_id].item()
                        # # 打印token及其概率
                        # print(f"{tokenizer.decode(generated_token_id)}: {generated_token_prob:.4f}")
                        if generated_token_id == tokenizer.eos_token_id:
                            break
                        completions_probs.append({tokenizer.decode(generated_token_id):generated_token_prob})

                    completions=[(generated_text, completions_probs)]
                    return completions
            
                else:
                    tokens = model.generate(
                            input_ids,
                            num_return_sequences=1,
                            max_length=input_ids.shape[1] + 300,
                            use_cache=True,
                            do_sample=False,
                        )

            for i in tokens:
                text = tokenizer.decode(i, skip_special_tokens=True)
                completions.append(text.strip())
                
        except RuntimeError as e:
            logging.error(f"Could not sample from model: {e}")
    return completions #list(set(completions))

def initialize_openai(args):
    api_key = open(f"{args.openai_creds_dir}/openai_api_key.txt").read()
    openai.organization = open(
        f"{args.openai_creds_dir}/openai_organization_id.txt"
    ).read()
    openai.api_key = api_key

def sample_code_from_openai_model(args, prompt_text):
    output_strs = []
    start = time.time()

    arch_mapping = {
        "codex": "code-davinci-002",
        "gpt3": "text-davinci-001",
        "davinci-002": "text-davinci-002",
        "davinci-003": "text-davinci-003",
        "ada": "text-ada-001",
        "babbage": "text-babbage-001",
        "curie": "text-curie-001",
    }
    engine_name = arch_mapping[args.arch]

    for i in range(args.num_samples):
        while time.time() - start < args.max_request_time:
            try:
                response = openai.Completion.create(
                    engine=engine_name,
                    prompt=prompt_text,
                    max_tokens=300,
                    n=1,
                    temperature=args.temperature,
                )
                output_strs += [
                    prompt_text + choice["text"] for choice in response["choices"]
                ]
                break
            except Exception as e:
                print(
                    f"Unexpected exception in generating solution. Sleeping again: {e}"
                )
                time.sleep(args.sleep_time)
    return output_strs

def write_jsonl(data, output_filepath):
    with open(output_filepath, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

def load_model_tokenizer(args, arch, model_path, model_dir, device):
    if arch in ["gpt3", "codex"]:
        initialize_openai(args)
        generate_code_fn = sample_code_from_openai_model
    elif "codegen" in arch:
        base_model = AutoModelForCausalLM.from_pretrained(
            f"Salesforce/{args.arch}", cache_dir="/data3/public_checkpoints/huggingface_models/codegen-6b-multi", low_cpu_mem_usage=True, torch_dtype=torch.float32,
            device_map={"": args.gpu[0]}
        )
        if args.model_path:
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
        else:
            model = base_model         

        tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/{args.arch}", cache_dir="/data3/public_checkpoints/huggingface_models/codegen-6b-multi", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        generate_code_fn = lambda args, prompt: sample_code_from_llm(
            args, prompt, model, tokenizer, device, args.return_probs
        )

    elif "Llama" in arch:
        base_model = AutoModelForCausalLM.from_pretrained(
            f"{model_dir}",
            low_cpu_mem_usage=True,
            #revision="float32",
            #torch_dtype=torch.float32,
            device_map={"": args.gpu[0]}
        )

        base_model = LlamaForCausalLM.from_pretrained(f"/data3/public_checkpoints/huggingface_models/Llama-2-7b-hf",
                                                         revision="float16",
                                                         torch_dtype=torch.float16,
                                                         low_cpu_mem_usage=True,
                                                     ).cuda()
        
        if args.model_path:
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            #ToDo
            #model = base_model
        else:
            model = base_model

        tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        generate_code_fn = lambda args, prompt: sample_code_from_llm(
            args, prompt, model, tokenizer, device, args.return_probs
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

        generate_code_fn = lambda args, prompt: sample_code_from_llm(
            args, prompt, model, tokenizer, device, args.return_probs
        )

    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            f"bigscience/{model_name}",
            # low_cpu_mem_usage=True,
            cache_dir="/data3/public_checkpoints/huggingface_models/bloom-7b",
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

        tokenizer = AutoTokenizer.from_pretrained(f"bigscience/{model_name}",cache_dir="/data3/public_checkpoints/huggingface_models/bloom-7b", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        generate_code_fn = lambda args, prompt: sample_code_from_llm(
            args, prompt, model, tokenizer, device, args.return_probs
        )

    return generate_code_fn

def generate_code_for_problems(args, except_tasks, output_filepath):
    if args.dataset == "MBPP":
        dataset = load_from_disk("datasets/mbpp_sanitized")  # Adjust based on actual path
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif args.dataset == "HumanEval":
        dataset = load_from_disk('datasets/humaneval')
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
        print(dataset.column_names)
    elif args.dataset == "GSM8K":
        dataset = load_from_disk('/home/zhangxuanming/DataLeakage_v2/dataset_gsm')
        print(dataset.column_names)

    # Handle the new format
    dataset = dataset["test"]  # Access the inner "test" dictionary

    if torch.cuda.is_available():  
        device = [f"cuda:{gpu}" for gpu in args.gpu]
    else:  
        device = torch.device("cpu")

    generate_code_fn = load_model_tokenizer(args, args.arch, args.model_path, args.model_dir, device)
    f = open(output_filepath, "a")

    for i in tqdm(range(len(dataset["task_id"]))):  # Iterate over the "test" dictionary
        if dataset["task_id"][i] in except_tasks:
            continue
        try:
            if args.dataset == "MBPP":
                prompt = format_prompt(
                    dataset["task_id"][i],
                    dataset["prompt"][i],
                    dataset["test"][i],  # Adjusted for the new format
                    dataset["canonical_solution"][i],  # Adjusted field names
                    args.num_use_cases,
                )
            else:
                # humaneval
                prompt = dataset["prompt"][i]
        except ValueError as e:
            logging.error(e)
            continue

        task_id = dataset["task_id"][i]

        if args.return_probs and not (args.temperature > 0):
            for completion in generate_code_fn(args, prompt):
                output = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": completion[0],
                    "prob": completion[1],
                }
                f.write(json.dumps(output) + "\n")
                f.flush()
        else:
            for completion in generate_code_fn(args, prompt):
                output = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": completion,
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
