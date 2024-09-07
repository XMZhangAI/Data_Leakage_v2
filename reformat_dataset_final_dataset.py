import json
from datasets import load_from_disk, concatenate_datasets, DatasetDict

input_path_original = "datasets/detect_dataset_all_v3/original_data_contamination_detection_dataset_truncate_epoch2"
input_path_varient = "datasets/detect_dataset_all_varient_v3/variant_data_contamination_detection_dataset_truncate_epoch2"

dataset_original = load_from_disk(input_path_original)['test']
dataset_varient = load_from_disk(input_path_varient)['test']

# 合并数据集
# dataset = concatenate_datasets([dataset_original, dataset_varient], split='test')
dataset = DatasetDict({'code generation original': dataset_original, 'code generation variant': dataset_varient})

dataset = dataset.remove_columns("prob")
dataset = dataset.rename_column("completion", "greedy_sample")
dataset = dataset.rename_column("completion_sample", "samples")
dataset = dataset.rename_column("task_id", "humaneval_task_id")
dataset = dataset.rename_column("id", "DETCON_id")
dataset = dataset.rename_column("leaked_data", "standard_solution")
dataset = dataset.rename_column("ground_truth_prob", "standard_solution_prob")
dataset.save_to_disk("datasets/DETCON")

print("hi")