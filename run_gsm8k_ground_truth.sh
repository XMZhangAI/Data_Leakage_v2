GPU=7
export CUDA_VISIBLE_DEVICES=${GPU}

Arch="Llama-2-7b-hf"
OutputDir="probs_Llama_2_7b_gsm8k_1k_ground_truth"
SaveModelPath="modelsave_llama_red_pajama_data_100K_gsm8k_reformat_1k/"


for i in {0..20}
do
    if [ $i -eq 20 ]; then
    python generate_lora_merge_ground_truth.py \
    --model-dir "/home/jiangxue/LLMs" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0 \
    --num-samples 1 \
    --acctual-num-samples 1 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset GSM8K \
    --model-path "" \
    --return_probs

    wait $!
    else
    python generate_lora_merge_ground_truth.py \
    --model-dir "/home/jiangxue/LLMs" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0 \
    --num-samples 1 \
    --acctual-num-samples 10 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset GSM8K \
    --model-path "/home/dongyh/DataLeakage/$SaveModelPath/checkpoint-epoch$i" \
    --return_probs

    wait $!
    fi

done

Arch="Mistral-7B-v0.1"
OutputDir="probs_Mistral_7B_gsm8k_1k_ground_truth"
SaveModelPath="modelsave_Mistral_red_pajama_data_100K_gsm8k_reformat_1k/"

for i in {0..20}
do
    if [ $i -eq 20 ]; then
    python generate_lora_merge_ground_truth.py \
    --model-dir "/home/jiangxue/LLMs" \
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
    python generate_lora_merge_ground_truth.py \
    --model-dir "/home/jiangxue/LLMs" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0 \
    --num-samples 1 \
    --acctual-num-samples 10 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset GSM8K \
    --model-path "/home/dongyh/DataLeakage/$SaveModelPath/checkpoint-epoch$i" \
    --return_probs

    wait $!
    fi

done