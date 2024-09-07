GPU=0
export CUDA_VISIBLE_DEVICES=${GPU}

Arch="Llama-2-7b-hf"
ModelPath="/home/jiangxue/LLMs/$Arch"
DatasetPath="datasets/red_pajama_data_100K_gsm8k_reformat_10000"
SaveModelPath="modelsave_llama_red_pajama_data_100K_gsm8k_reformat_1k/"
OutputDir="outputs_llama_red_pajama_data_100K_gsm8k_reformat_1k_ppl/"

# python leakage.py \
# --do_train \
# --save_strategy no \
# --num_train_epochs 20 \
# --learning_rate 2e-4 \
# --per_device_train_batch_size 1 \
# --gradient_accumulation_steps 32 \
# --logging_steps 1 \
# --output_dir $SaveModelPath \
# --save_total_limit 2 \
# --overwrite_output_dir || exit

for i in {0..20}
do
    # if [ $i -eq 20 ]; then
    # python generate_lora_merge.py \
    # --model-dir "/home/jiangxue/LLMs" \
    # --output-dir $OutputDir \
    # --arch $Arch \
    # --temperature 0 \
    # --num-samples 1 \
    # --acctual-num-samples 10 \
    # --output-file-suffix test \
    # --num-use-cases 0 \
    # --dataset GSM8K \
    # --model-path "" \
    # --return_probs

    # wait $!
    # else
    # python generate_lora_merge.py \
    # --model-dir "/home/jiangxue/LLMs" \
    # --output-dir $OutputDir \
    # --arch $Arch \
    # --temperature 0 \
    # --num-samples 1 \
    # --acctual-num-samples 10 \
    # --output-file-suffix test \
    # --num-use-cases 0 \
    # --dataset GSM8K \
    # --model-path "/home/dongyh/DataLeakage/$SaveModelPath/checkpoint-epoch$i" \
    # --return_probs

    # wait $!
    # fi

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
    --dataset GSM8K \
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
    --dataset GSM8K \
    --model-path "/home/dongyh/DataLeakage/$SaveModelPath/checkpoint-epoch$i" \
    --return_probs

    wait $!
    fi

done