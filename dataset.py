import json
import os
from datasets import Dataset, DatasetDict

def convert_to_humaneval_format(input_path, output_path):
    humaneval_data = []

    # Read the JSON file line by line
    with open(input_path, 'r') as file:
        for line in file:
            if line.strip():  # Ignore empty lines
                try:
                    item = json.loads(line)
                    humaneval_item = {
                        'task_id': item.get('task_id'),
                        'prompt': item.get('prompt'),
                        'canonical_solution': item.get('canonical_solution'),
                        'test': item.get('test'),
                        'entry_point': item.get('entry_point')
                    }
                    humaneval_data.append(humaneval_item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

    # Create a Dataset
    new_dataset = Dataset.from_list(humaneval_data)

    # Wrap the dataset in a DatasetDict
    dataset_dict = DatasetDict({'test': new_dataset})

    # Save the dataset in the Arrow format
    dataset_dict.save_to_disk(output_path)

if __name__ == "__main__":
    input_path = "/home/zhangxuanming/DataLeakage_v2/datasets/multi_leakage_source_gsm.json"
    output_path = "/home/zhangxuanming/DataLeakage_v2/dataset_gsm"

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Convert and save the dataset
    convert_to_humaneval_format(input_path, output_path)

    print(f"Dataset successfully saved to {output_path}")
