torchrun --nproc_per_node 8 --master_port 29500 generate_with_aug.py \
    --model-path /home/data/wyy/checkpoints/llava-v1.5-13b/ \
    --image_file_list /home/data/wyy/projects/SeVa/llava_instruction_filtered.json \
    --image_path /home/data/wyy/datasets/VLFeedback/merged_images/ \
    --save_dir /home/data/wyy/projects/SeVa/ \
    --res_file "llava_instruction_filtered_diffusion_step800_13b.jsonl" \
    --augmentation "diffusion" \
    --noise_step 800 \


