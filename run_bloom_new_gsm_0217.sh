GPU=3
export CUDA_VISIBLE_DEVICES=${GPU}

Arch="bloom-7b1"
ModelPath="bigscience/bloom-7b1"
DatasetPath="/home/zhangxuanming/DataLeakage_v2/datasets/red_pajama_data_100K_gsm8k_reformat_10000"
SaveModelPath="modelsave_bloom-7B_new_gsm_10k_0217"
OutputDir="outputs_bloom-7B_new_gsm_10k_0217"

python3 leakage_v3.py \
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

for i in {0..20}
do
    if [ $i -eq 20 ]; then
    python3 generate_lora_merge_v4.py \
    --model-dir "bigscience/bloom-7b1" \
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
    --model-dir "bigscience/bloom-7b1" \
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
    --model-dir "bigscience/bloom-7b1" \
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
    --model-dir "bigscience/bloom-7b1" \
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
