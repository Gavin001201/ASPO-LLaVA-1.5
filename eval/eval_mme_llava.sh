batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data-perplexity-13b-3

                python seva/model_vqa_loader.py \
                    --model-path ./checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-13b \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/MME/llava_mme.jsonl \
                    --image-folder /home/data/wyy/projects/SeVa/seva/playground/data/eval/MME/MME_Benchmark_release_version \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/MME/answers/${MODEL_VERSION}/${MODEL_VERSION}.jsonl \
                    --temperature 0 \
                    --conv-mode vicuna_v1

                cd /home/data/wyy/projects/SeVa/seva/playground/data/eval/MME

                python convert_answer_to_mme.py --experiment ${MODEL_VERSION}

                cd /home/data/wyy/projects/SeVa/seva/playground/data/eval/MME/eval_tool

                python calculation.py --results_dir /home/data/wyy/projects/SeVa/seva/playground/data/eval/MME/answers/${MODEL_VERSION} > /home/data/wyy/projects/SeVa/seva/playground/data/eval/MME/answers/${MODEL_VERSION}/eval.log

                cd /home/data/wyy/projects/SeVa

            done
        done
    done
done