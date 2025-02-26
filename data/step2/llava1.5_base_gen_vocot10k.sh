torchrun --nproc_per_node 8 --master_port 29502 generate_with_base_model.py \
    --model-path /home/data/wyy/checkpoints/llava-v1.5-7b \
    --image_file_list /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train/vocot_dpo_10000_chosen_integrated_4100_4100_1800.json \
    --image_path /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train/VoCoT \
    --save_dir /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train \
    --res_file "vocot_answer_file_10k_llava.jsonl" \


