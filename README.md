# DataLeakage

# 构造多数据源泄露
reformat_data.py 

Note:
1. 需要修改泄漏数据的路径为dataset/multi_leakage_source_*.json 
2. multi_leakage_source_humaneval.json中的completion可能存在函数签名，如果使用，需要删除后再与函数签名concat。final为完整代码。
3. 可能需要将json转化为dataset，可以读取json后，运行如下代码进行转换：

new_dataset = Dataset.from_list(new_dataset)
new_dataset = DatasetDict({'test': new_dataset})
new_dataset.save_to_disk(save_path)

# 模拟数据泄漏
leakage_v2.py

python3 leakage_v2.py \
--do_train \
--save_strategy no \
--num_train_epochs 20 \
--learning_rate 2e-4 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32 \
--logging_steps 1 \
--output_dir $SaveModelPath \
--save_total_limit 2 \
--overwrite_output_dir || exit

Example: 
run_codellama_new_humaneval_0217.sh 
run_codegen_new_humaneval_0217.sh
run_ground_truth_probs_codegen.sh
run_ground_truth_probs_codellama.sh

Note:
- 修改构造的多数据源泄露训练数据的路径（DatasetPath）和模型存储路径（SaveModelPath）以及模型采样结果路径（OutputDir）

# truncate采样结果
truncate_code.py

Note:
- 修改采样结果的文件夹名，截断的结果再./truncat下

# 构造泄露检测数据集
construct_original_data_contamination_detection_dataset_truncate_iteration.py
construct_variant_data_contamination_detection_dataset_truncate_iteration.py
reformat_dataset_final_dataset.py

Note:
- 修改文件夹路径

# CDD
CDD.py
eval/baselines.py

Note:
- 修改数据集路径

# TED
TED.py 


