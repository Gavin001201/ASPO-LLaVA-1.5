#!/bin/bash
batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data-dpo-13b

                python ../seva/llava/eval/model_vqa_science.py \
                    --model-path ../checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-13b \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/scienceqa/llava_test_CQM-A.json \
                    --image-folder /home/data/wyy/projects/SeVa/seva/playground/data/eval/scienceqa/images/test \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/scienceqa/answers/${MODEL_VERSION}.jsonl \
                    --single-pred-prompt \
                    --temperature 0 \
                    --conv-mode vicuna_v1

                python ../seva/llava/eval/eval_science_qa.py \
                    --base-dir /home/data/wyy/projects/SeVa/seva/playground/data/eval/scienceqa \
                    --result-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/scienceqa/answers/${MODEL_VERSION}.jsonl \
                    --output-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/scienceqa/answers/${MODEL_VERSION}_output.jsonl \
                    --output-result /home/data/wyy/projects/SeVa/seva/playground/data/eval/scienceqa/answers/${MODEL_VERSION}_result.json

            done
        done
    done
done