GPU=3
export CUDA_VISIBLE_DEVICES=${GPU}

Arch="CodeLlama-7b-hf"
OutputDir="probs_codellama_7b_10k_ground_truth_3"
SaveModelPath="modelsave_CodeLlama7B_new_humaneval_10k_0217"

for i in {0..20}
do
    if [ $i -eq 20 ]; then
    python3 generate_lora_merge_ground_truth.py \
    --model-dir "/data3/public_checkpoints/huggingface_models/CodeLlama-7b-hf/models--codellama--CodeLlama-7b-hf/snapshots/6c284d1468fe6c413cf56183e69b194dcfa27fe6" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0 \
    --num-samples 1 \
    --acctual-num-samples 1 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset HumanEval \
    --model-path "" \
    --return_probs

    wait $!
    else
    python3 generate_lora_merge_ground_truth.py \
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

done

