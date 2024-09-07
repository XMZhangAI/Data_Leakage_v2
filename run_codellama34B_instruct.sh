GPU=2
export CUDA_VISIBLE_DEVICES=${GPU}

Arch="CodeLlama-7b-hf"
ModelPath="/home/jiangxue/LLMs/$Arch"
DatasetPath="datasets/starcoder_data_reformat_1K"
SaveModelPath="modelsave_codellama_7b_1k_lr1e_3_2/"
OutputDir="outputs_codellama_7b_1k_lr1e_3_2/"

for i in {0..19}
do
    if [ $i -eq 20 ]; then
    python generate_lora_merge.py \
    --model-dir "/home/jiangxue/LLMs" \
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
    python generate_lora_merge.py \
    --model-dir "/home/jiangxue/LLMs" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0 \
    --num-samples 1 \
    --acctual-num-samples 10 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset HumanEval \
    --model-path "/home/dongyh/DataLeakage/$SaveModelPath/checkpoint-epoch$i" \
    --return_probs

    wait $!
    fi

    if [ $i -eq 20 ]; then
    python generate_lora_merge.py \
    --model-dir "/home/jiangxue/LLMs" \
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
    python generate_lora_merge.py \
    --model-dir "/home/jiangxue/LLMs" \
    --output-dir $OutputDir \
    --arch $Arch \
    --temperature 0.8 \
    --num-samples 50 \
    --acctual-num-samples 10 \
    --output-file-suffix test \
    --num-use-cases 0 \
    --dataset HumanEval \
    --model-path "/home/dongyh/DataLeakage/$SaveModelPath/checkpoint-epoch$i" \
    --return_probs

    wait $!
    fi

done