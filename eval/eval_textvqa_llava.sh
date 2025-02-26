#!/bin/bash
batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data-loglikelihood-7b-token
                python seva/model_vqa_loader.py \
                    --model-path ./checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
                    --image-folder /home/data/wyy/projects/SeVa/seva/playground/data/eval/textvqa/train_images \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/textvqa/answers/${MODEL_VERSION}.jsonl \
                    --temperature 0 \
                    --conv-mode vicuna_v1

                python seva/llava/eval/eval_textvqa.py \
                    --annotation-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
                    --result-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/textvqa/answers/${MODEL_VERSION}.jsonl

            done
        done
    done
done