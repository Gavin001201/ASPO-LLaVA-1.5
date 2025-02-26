#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="/home/data/wyy/projects/SeVa/seva/playground/data/eval/gqa/data"

batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data-dpo-13b

                for IDX in $(seq 0 $((CHUNKS-1))); do
                    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ../seva/model_vqa_loader.py \
                        --model-path ../checkpoints/${MODEL_VERSION} \
                        --model-base /home/data/wyy/checkpoints/llava-v1.5-13b \
                        --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/gqa/$SPLIT.jsonl \
                        --image-folder /home/data/wyy/projects/SeVa/seva/playground/data/eval/gqa/data/images \
                        --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/gqa/answers/$SPLIT/$MODEL_VERSION/${CHUNKS}_${IDX}.jsonl \
                        --num-chunks $CHUNKS \
                        --chunk-idx $IDX \
                        --temperature 0 \
                        --conv-mode vicuna_v1 &
                done

                wait

                output_file=/home/data/wyy/projects/SeVa/seva/playground/data/eval/gqa/answers/$SPLIT/$MODEL_VERSION/merge.jsonl

                # Clear out the output file if it exists.
                > "$output_file"

                # Loop through the indices and concatenate each file.
                for IDX in $(seq 0 $((CHUNKS-1))); do
                    cat /home/data/wyy/projects/SeVa/seva/playground/data/eval/gqa/answers/$SPLIT/$MODEL_VERSION/${CHUNKS}_${IDX}.jsonl >> "$output_file"
                done

                python ../seva/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

                cd $GQADIR
                python eval.py --tier testdev_balanced

            done
        done
    done
done