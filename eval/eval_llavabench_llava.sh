batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_10-lr_2e-6-lora_r_1024-scaling_factor_1-new_data6

                # python ../seva/llavabench_eval.py \
                #     --model-path ../checkpoints/${MODEL_VERSION} \
                #     --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
                #     --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
                #     --image-folder /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/images \
                #     --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/answers/${MODEL_VERSION}.jsonl \
                #     --temperature 0 \
                #     --conv-mode vicuna_v1

                cd /home/data/wyy/projects/SeVa/seva/llava/eval

                # python eval_gpt_review_bench.py \
                #     --question /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
                #     --context /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/context.jsonl \
                #     --rule /home/data/wyy/projects/SeVa/seva/llava/eval/table/rule.json \
                #     --answer-list \
                #         /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
                #         /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/answers/${MODEL_VERSION}.jsonl \
                #     --output \
                #         /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/reviews/${MODEL_VERSION}.jsonl

                python summarize_gpt_review.py -f /home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/reviews/${MODEL_VERSION}.jsonl

            done
        done
    done
done