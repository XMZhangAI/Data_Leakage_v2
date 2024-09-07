GPU=3
export CUDA_VISIBLE_DEVICES=${GPU}

Arch="Llama-2-7b-hf"
ModelPath="/data3/public_checkpoints/huggingface_models/Llama-2-7b-hf"
DatasetPath="/home/zhangxuanming/DataLeakage_v2/datasets/red_pajama_data_100K_gsm8k_reformat_10000"
SaveModelPath="modelsave_llama2-7B_new_gsm_10k_0217"
OutputDir="outputs_llama2-7B_new_gsm_10k_0217"

for i in {0..20}
do
    if [ $i -eq 20 ]; then
    python3 generate_lora_merge_v4.py \
    --model-dir "/data3/public_checkpoints/huggingface_models/Llama-2-7b-hf" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0 \
    --num-samples 1 \
    --acctual-num-samples 10 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset GSM8K \
    --model-path "" \
    --return_probs

    wait $!
    else
    python3 generate_lora_merge_v4.py \
    --model-dir "/data3/public_checkpoints/huggingface_models/Llama-2-7b-hf" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0 \
    --num-samples 1 \
    --acctual-num-samples 10 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset GSM8K \
    --model-path "/home/zhangxuanming/DataLeakage_v2/$SaveModelPath/checkpoint-epoch$i" \
    --return_probs

    wait $!
    fi

    if [ $i -eq 20 ]; then
    python3 generate_lora_merge_v4.py \
    --model-dir "/data3/public_checkpoints/huggingface_models/Llama-2-7b-hf" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0.8 \
    --num-samples 50 \
    --acctual-num-samples 10 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset GSM8K \
    --model-path "" \
    --return_probs

    wait $!
    else
    python3 generate_lora_merge_v4.py \
    --model-dir "/data3/public_checkpoints/huggingface_models/Llama-2-7b-hf" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0.8 \
    --num-samples 50 \
    --acctual-num-samples 10 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset GSM8K \
    --model-path "/home/zhangxuanming/DataLeakage_v2/$SaveModelPath/checkpoint-epoch$i" \
    --return_probs

    wait $!
    fi

done
