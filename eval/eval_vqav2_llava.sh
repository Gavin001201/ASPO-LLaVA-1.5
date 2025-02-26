#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do

            MODEL_VERSION=hyparameter_search_bz_$bz-lr_$lr-lora_r_$lora_r-text_reward57

            for IDX in $(seq 0 $((CHUNKS-1))); do
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python seva/model_vqa_loader.py \
                    --model-path ./checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/vqav2/$SPLIT.jsonl \
                    --image-folder /home/data/wyy/projects/SeVa/seva/playground/data/eval/vqav2/test2015 \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/vqav2/answers/$SPLIT/$MODEL_VERSION/${CHUNKS}_${IDX}.jsonl \
                    --num-chunks $CHUNKS \
                    --chunk-idx $IDX \
                    --temperature 0 \
                    --conv-mode vicuna_v1 &
            done

            wait

            output_file=/home/data/wyy/projects/SeVa/seva/playground/data/eval/vqav2/answers/$SPLIT/$MODEL_VERSION/merge.jsonl

            # Clear out the output file if it exists.
            > "$output_file"

            # Loop through the indices and concatenate each file.
            for IDX in $(seq 0 $((CHUNKS-1))); do
                cat /home/data/wyy/projects/SeVa/seva/playground/data/eval/vqav2/answers/$SPLIT/$MODEL_VERSION/${CHUNKS}_${IDX}.jsonl >> "$output_file"
            done

            python seva/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $MODEL_VERSION

        done
    done
done