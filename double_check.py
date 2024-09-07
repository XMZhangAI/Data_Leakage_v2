import json
import os
from datasets import load_from_disk

def convert_to_json_format(dataset_path, output_path):
    # 加载数据集
    dataset = load_from_disk(dataset_path)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 打开输出文件
    with open(output_path, 'w') as file:
        for example in dataset:
            try:
                json.dump(example, file)
                file.write('\n')  # 每个 JSON 对象占一行
            except (TypeError, IOError) as e:
                print(f"Error writing JSON: {e}")
                continue

if __name__ == "__main__":
    dataset_path = "/home/zhangxuanming/DataLeakage_v2/datasets/starcoder_data_100K_new_humaneval_reformat_test_10000"
    output_path = "/home/zhangxuanming/DataLeakage_v2/output2.json"

    # 转换并保存 JSON 文件
    convert_to_json_format(dataset_path, output_path)

    print(f"JSON successfully saved to {output_path}")
