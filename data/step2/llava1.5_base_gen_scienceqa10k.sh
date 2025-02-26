torchrun --nproc_per_node 8 --master_port 29500 generate_with_base_model.py \
    --model-path /home/data/wyy/checkpoints/llava-v1.5-7b \
    --image_file_list /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train/scienceqa_dpo_5381_chosen_train_single_image.json \
    --image_path /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train/ScienceQA \
    --save_dir /home/data/wyy/projects/SeVa/data/custom_cot_dpo_train \
    --res_file "scienceqa_answer_file_5k_llava.jsonl" \


