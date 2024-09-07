GPU=2
export CUDA_VISIBLE_DEVICES=${GPU}

Arch="codegen-6B-multi"
OutputDir="probs_codegen_6b_1k_ground_truth_3"
SaveModelPath="modelsave_codegen6B_1k"

python generate_lora_merge_ground_truth.py \
--model-dir "/home/jiangxue/LLMs" \
--output-dir $OutputDir \
--arch $Arch \
--temperature 0 \
--num-samples 1 \
--acctual-num-samples 10 \
--output-file-suffix test \
--num-use-cases 0 \
--dataset HumanEval \
--model-path "/home/dongyh/DataLeakage/$SaveModelPath/checkpoint-epoch18" \
--return_probs