batch_size=(10)
learning_rates=(2e-6)
lora_rs=(1024)
scaling_factors=(1)

for bz in "${batch_size[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for scaling_factor in "${scaling_factors[@]}"; do

                MODEL_VERSION=bz_$bz-lr_$lr-lora_r_$lora_r-scaling_factor_$scaling_factor-new_data-dpo-13b

                torchrun --nproc_per_node 8 --master_port 29501 ../seva/pope_eval.py \
                    --coco_path /home/data/wyy/datasets/coco2014 \
                    --pope_path ../data/POPE/ \
                    --model-path ../checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-13b \
                    --save_dir ../seva/pope_result/${MODEL_VERSION} \
                    --set random


                torchrun --nproc_per_node 8 --master_port 29501 ../seva/pope_eval.py \
                    --coco_path /home/data/wyy/datasets/coco2014 \
                    --pope_path ../data/POPE/ \
                    --model-path ../checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-13b \
                    --save_dir ../seva/pope_result/${MODEL_VERSION} \
                    --set popular


                torchrun --nproc_per_node 8 --master_port 29501 ../seva/pope_eval.py \
                    --coco_path /home/data/wyy/datasets/coco2014 \
                    --pope_path ../data/POPE/ \
                    --model-path ../checkpoints/${MODEL_VERSION} \
                    --model-base /home/data/wyy/checkpoints/llava-v1.5-13b \
                    --save_dir ../seva/pope_result/${MODEL_VERSION} \
                    --set adv

                python ../seva/pope_calculate.py --path ../seva/pope_result/${MODEL_VERSION} > ../seva/pope_result/${MODEL_VERSION}/eval.log

            done
        done
    done
done