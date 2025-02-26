torchrun --nproc_per_node 8 --master_port 29502 generate_with_aug.py \
    --model-path /home/data/wyy/projects/SeVa/checkpoints/bz_10-lr_2e-6-lora_r_1024-scaling_factor_1-new_data2 \
    --model-base /home/data/wyy/checkpoints/llava-v1.5-7b \
    --image_file_list /home/data/wyy/projects/SeVa/llava_instruction_filtered.json \
    --image_path /home/data/wyy/datasets/VLFeedback/merged_images/ \
    --save_dir /home/data/wyy/projects/SeVa/ \
    --res_file "llava_instruction_generated_chosen_7b_iter_1.jsonl" \


