#!/bin/bash
batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data1

                python seva/playground/data/eval/aro/aro.py \
                    --model-path ./checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
                    --image-folder /home/data/wyy/datasets/vg/VG_100K \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/aro/parsed_vg_att.jsonl \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/aro/answers/${MODEL_VERSION}/vg_att.jsonl \
                    --temperature 0 \
                    --conv-mode vicuna_v1

                python seva/playground/data/eval/aro/aro.py \
                    --model-path ./checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
                    --image-folder /home/data/wyy/datasets/vg/VG_100K \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/aro/parsed_vg_rel.jsonl \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/aro/answers/${MODEL_VERSION}/vg_rel.jsonl \
                    --temperature 0 \
                    --conv-mode vicuna_v1

                python seva/playground/data/eval/aro/eval.py \
                    --annotation-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/aro/answers/${MODEL_VERSION}/vg_att.jsonl \

                python seva/playground/data/eval/aro/eval.py \
                    --annotation-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/aro/answers/${MODEL_VERSION}/vg_rel.jsonl \

            done
        done
    done
done