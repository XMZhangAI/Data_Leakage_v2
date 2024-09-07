import os
# 设置 Hugging Face 镜像地址
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定模型保存路径
model_dir = "/data3/public_checkpoints/huggingface_models/codegen25-7b-mono"

# 创建路径
os.makedirs(model_dir, exist_ok=True)

# 指定模型名称
model_name = "Salesforce/codegen25-7b-mono_P"

# 直接将模型和tokenizer下载并保存到指定路径
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)

# 保存模型和tokenizer到指定路径（这一步是冗余的，因为上一步已经将文件下载到 model_dir 了）
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
