Arch="CodeLlama-7b-hf"
ModelPath="/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6"
DatasetPath="datasets/starcoder_data_100K_new_humaneval_reformat_test_10000"
SaveModelPath="modelsave_CodeLlama7B_new_humaneval_10k_0217"
OutputDir="outputs_CodeLlama7B_new_humaneval_10k_0217"

for i in {0..20}
do
    if [ $i -eq 20 ]; then
        python3 generate_lora_merge_v3.py \
        --model-dir "/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6" \
        --output-dir $OutputDir \
        --arch $Arch \
        --temperature 0 \
        --num-samples 1 \
        --acctual-num-samples 10 \
        --output-file-suffix test \
        --num-use-cases 0 \
        --dataset HumanEval \
        --model-path "" \
        --return_probs

    wait $!
    else
        python3 generate_lora_merge_v3.py \
        --model-dir "/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6" \
        --output-dir $OutputDir \
        --arch $Arch \
        --temperature 0 \
        --num-samples 1 \
        --acctual-num-samples 10 \
        --output-file-suffix test \
        --num-use-cases 0 \
        --dataset HumanEval \
        --model-path "/home/zhangxuanming/DataLeakage_v2/$SaveModelPath/checkpoint-epoch$i" \
        --return_probs

    wait $!
    fi

    if [ $i -eq 20 ]; then
        python3 generate_lora_merge_v3.py \
        --model-dir "/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6" \
        --output-dir $OutputDir \
        --arch $Arch \
        --temperature 0.8 \
        --num-samples 50 \
        --acctual-num-samples 10 \
        --output-file-suffix test \
        --num-use-cases 0 \
        --dataset HumanEval \
        --model-path "" \
        --return_probs

    wait $!
    else
        python3 generate_lora_merge_v3.py \
        --model-dir "/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6" \
        --output-dir $OutputDir \
        --arch $Arch \
        --temperature 0.8 \
        --num-samples 50 \
        --acctual-num-samples 10 \
        --output-file-suffix test \
        --num-use-cases 0 \
        --dataset HumanEval \
        --model-path "/home/zhangxuanming/DataLeakage_v2/$SaveModelPath/checkpoint-epoch$i" \
        --return_probs

    wait $!
    fi

done
