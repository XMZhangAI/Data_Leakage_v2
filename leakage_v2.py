import os, math
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch

# import argparse

# parser_1 = argparse.ArgumentParser()
# parser_1.add_argument('--model_name', type=str, default='CodeLlama-7b-hf')
# #CodeLlama-7b-hf /checkpoints/codegen-6B-multi #Llama-2-7b-hf
# parser_1.add_argument('--model_path', type=str, default='/home/jiangxue/LLMs')
# #f"/home/jiangxue/LLMs/{model_name}" # Salesforce/codegen-6B-multi
# parser_1.add_argument('--dataset_path', type=str, default='datasets/starcoder_data_100K_new_humaneval_reformat_1000')
# parser_1.add_argument('--max_seq_length', type=int, default=1024) 
# parser_1.add_argument('--lora_rank', type=int, default=128)
# args = parser_1.parse_args()

contain = True

model_name = "Llama-2-7b-hf"
model_path = f"/data3/public_checkpoints/huggingface_models/Llama-2-7b-hf"
dataset_path = "/home/zhangxuanming/DataLeakage_v2/datasets/red_pajama_data_100K_gsm8k_reformat_10000"
last_checkpoint = None
max_seq_length = 1024
lora_rank = 128

# model_name = args.model_name
# model_path = args.model_path
# dataset_path = args.dataset_path
# last_checkpoint = None
# max_seq_length = args.max_seq_length
# lora_rank = args.lora_rank

if contain:
    # dataset = load_from_disk('datasets/starcoder_data_reformat_1K')
    dataset = load_from_disk(dataset_path)
else:
    dataset = load_from_disk('datasets/humaneval')['test']

if "Llama" in model_name:
    if last_checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(
            last_checkpoint
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            f"{model_path}"
        )
    model = model.to('cuda', dtype=torch.float32)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
elif "codegen" in model_name:
        if last_checkpoint is not None:
            model = AutoModelForCausalLM.from_pretrained(
                last_checkpoint,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                f"Salesforce/{model_name}",
                # low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        ## IMPORTANT: DO NOT REMOVE
        model = model.to('cuda', dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/{model_name}", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

elif "Mistral" in model_name:
        if last_checkpoint is not None:
            model = AutoModelForCausalLM.from_pretrained(
                last_checkpoint,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                f"/home/dongyh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
                # low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        ## IMPORTANT: DO NOT REMOVE
        model = model.to('cuda', dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(f"/home/dongyh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

else:
    raise ValueError(
        f"{model_name} is not a valid model name or path."
    )
    
from peft import LoraConfig, PeftModel, get_peft_model

# FIXME
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
    )

model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())
    



from transformers import TrainerCallback
class SaveModelCallback(TrainerCallback):
    """
    Callback that saves the model after each epoch
    """
    def on_epoch_end(self, args, state, control, model, **kwargs):
        """
        Save the model at the end of each epoch
        """
            
        if math.floor(state.epoch)< 100:
            save_path = os.path.join(args.output_dir, 'checkpoint-epoch{:.2f}'.format(round(state.epoch)))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_path)


def truncate(ex, tokenizer, max_length):
    return tokenizer.decode(
        tokenizer(ex, max_length=max_length, truncation=True,).input_ids
    )

prompt_column = 'prompt'
completion_column = 'canonical_solution'
def preprocess_example(example):
    input_str = example[prompt_column]
    output_str = example[completion_column]
    if input_str:
        input_token_ids = tokenizer.encode(input_str, verbose=False) 
    else:
        input_token_ids = []
    # target_token_ids = tokenizer.encode(output_str, add_special_tokens= False, verbose=False) + [tokenizer.eos_token_id]
    # input_ids = input_token_ids + target_token_ids 
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(input_str+output_str, add_special_tokens= False, verbose=False) + [tokenizer.eos_token_id]
    labels_input_ids = ([-100] * len(input_token_ids)) + input_ids[len(input_token_ids):]

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels_input_ids = labels_input_ids[:max_seq_length]
    return {
        "input_ids": torch.IntTensor(input_ids).cuda(),
        "labels": torch.IntTensor(labels_input_ids).cuda(),
    }


# Data collator
parser = HfArgumentParser((Seq2SeqTrainingArguments))
training_args = parser.parse_args_into_dataclasses()[0]
print(training_args)
label_pad_token_id = -100
fp16 = False
train_dataset = dataset
max_train_samples = None

with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_dataset.map(
        preprocess_example,
        # remove_columns=column_names,
    )
if max_train_samples is not None:
    # Number of samples might increase during Feature Creation, We select only specified max samples
    max_train_samples = min(len(train_dataset),max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples))

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

# FIXME:
# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[SaveModelCallback()],
)

old_collator = trainer.data_collator
trainer.data_collator = lambda data: dict(old_collator(data))

# Training
if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        max_train_samples
        if max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
