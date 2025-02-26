torchrun --nproc_per_node 8 --master_port 29502 generate_with_base_model.py \
    --model-path /home/data/wyy/checkpoints/llava-v1.5-7b \
    --image_file_list /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train/aokvqa_dpo_10000_chosen.json \
    --image_path /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train/AOKVQA/train2017 \
    --save_dir /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train \
    --res_file "aokvqa_answer_file_10k_llava_test.jsonl" \


