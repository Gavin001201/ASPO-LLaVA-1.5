#!/bin/bash
SPLIT="MMBench_DEV_CN_legacy"
language="cn"

batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data6

                python ../seva/mmbench_eval.py \
                    --model-path ../checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/mmbench/$SPLIT.tsv \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/mmbench/answers_cn/$SPLIT/${MODEL_VERSION}.jsonl \
                    --single-pred-prompt \
                    --temperature 0 \
                    --lang $language \
                    --conv-mode vicuna_v1

                cd /home/data/wyy/projects/SeVa/seva/playground/data/eval/mmbench

                python convert_mmbench_for_submission.py \
                    --annotation-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/mmbench/$SPLIT.tsv \
                    --result-dir /home/data/wyy/projects/SeVa/seva/playground/data/eval/mmbench/answers_cn/$SPLIT \
                    --upload-dir /home/data/wyy/projects/SeVa/seva/playground/data/eval/mmbench/answers_upload_cn/$SPLIT \
                    --experiment ${MODEL_VERSION}

            done
        done
    done
done