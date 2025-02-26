# MODEL_VERSION=llava_lora_ft_cot_dpo_r1024_a2048_test824

batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data-dpo-13b

                TEXT_DPO_DATA=/home/data/wyy/projects/SeVa/data/llava_instruction_diffusion_step500_split_clip_score_13b.json

                deepspeed SeVa/seva/train_dpo_ours.py \
                    --lora_enable True --lora_r $lora_r --lora_alpha $[lora_r*2] --mm_projector_lr 0 \
                    --deepspeed /home/data/wyy/projects/SeVa/seva/scripts/zero3.json \
                    --model_name_or_path /home/data/wyy/checkpoints/llava-v1.5-13b \
                    --version v1 \
                    --textvqa_data_path ${TEXT_DPO_DATA} \
                    --textvqa_image_path /home/data/wyy/datasets/VLFeedback/merged_images/  \
                    --vision_tower /home/data/wyy/checkpoints/clip-vit-large-patch14-336 \
                    --mm_projector_type mlp2x_gelu \
                    --mm_vision_select_layer -2 \
                    --mm_use_im_start_end False \
                    --mm_use_im_patch_token False \
                    --scaling_factor $scaling_factor \
                    --image_aspect_ratio pad \
                    --group_by_modality_length True \
                    --bf16 True \
                    --output_dir ../checkpoints/${MODEL_VERSION} \
                    --num_train_epochs 1 \
                    --per_device_train_batch_size $bz \
                    --per_device_eval_batch_size 4 \
                    --gradient_accumulation_steps 1 \
                    --evaluation_strategy "no" \
                    --save_strategy "steps" \
                    --save_steps 50000 \
                    --save_total_limit 1 \
                    --learning_rate $lr \
                    --weight_decay 0. \
                    --warmup_steps 0 \
                    --lr_scheduler_type "cosine" \
                    --logging_steps 1 \
                    --tf32 True \
                    --model_max_length 2048 \
                    --gradient_checkpointing True \
                    --dataloader_num_workers 4 \
                    --lazy_preprocess True \
                    --report_to wandb \
                    --run_name ${MODEL_VERSION} \
                    --beta 0.1

            done
        done
    done
done