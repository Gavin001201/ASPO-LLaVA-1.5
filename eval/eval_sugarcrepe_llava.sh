#!/bin/bash
batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=seva-7b-diffu800

                python seva/playground/data/eval/sugarcrepe/sugar.py \
                    --model-path ./checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
                    --image-folder /home/data/wyy/datasets/coco2017/val2017 \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/sugarcrepe/parsed_sugar.jsonl \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/sugarcrepe/answers/${MODEL_VERSION}.jsonl \
                    --temperature 0 \
                    --conv-mode vicuna_v1

                python seva/playground/data/eval/sugarcrepe/eval.py \
                    --annotation-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/sugarcrepe/answers/${MODEL_VERSION}.jsonl \

            done
        done
    done
done