batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data6

                python ../seva/model_vqa.py \
                    --model-path ../checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
                    --question-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
                    --image-folder /home/data/wyy/projects/SeVa/seva/playground/data/eval/mm-vet/mm-vet/images \
                    --answers-file /home/data/wyy/projects/SeVa/seva/playground/data/eval/mm-vet/answers/${MODEL_VERSION}.jsonl \
                    --temperature 0 \
                    --conv-mode vicuna_v1

                mkdir -p /home/data/wyy/projects/SeVa/seva/playground/data/eval/mm-vet/results

                python ../seva/scripts/convert_mmvet_for_eval.py \
                    --src /home/data/wyy/projects/SeVa/seva/playground/data/eval/mm-vet/answers/${MODEL_VERSION}.jsonl \
                    --dst /home/data/wyy/projects/SeVa/seva/playground/data/eval/mm-vet/results/${MODEL_VERSION}.json

            done
        done
    done
done