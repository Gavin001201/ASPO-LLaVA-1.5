import json
import random
import math
from tqdm import tqdm

def softmax(x):
    exp_x = [math.exp(i) for i in x]
    sum_exp_x = sum(exp_x)
    return [i / sum_exp_x for i in exp_x]

mdpo_path = '/home/data/wyy/projects/SeVa/mdpo_llava_10k_resplit.json'
target_path = '/home/data/wyy/projects/SeVa/mdpo_llava_10k_resplit_min_max_normalize.json'
    
    
with open(mdpo_path, mode='r', encoding='utf-8') as file:
    mdpo_dataset=json.load(file)

for data in mdpo_dataset:
    chosen_clip_score = data['chosen_clip_score']
    reject_clip_score = data['reject_clip_score']
    
    # data['chosen_clip_score'] = softmax(chosen_clip_score)
    # 计算 Min-Max 归一化
    chosen_min = min(chosen_clip_score)
    chosen_max = max(chosen_clip_score)
    chosen_mean = sum(chosen_clip_score) / len(chosen_clip_score)
    reject_min = min(reject_clip_score)
    reject_max = max(reject_clip_score)
    reject_mean = sum(reject_clip_score) / len(reject_clip_score)
    
    data['chosen_clip_score'] = [(item - chosen_min) / (chosen_max - chosen_min + 1e-6) for item in chosen_clip_score]
    data['reject_clip_score'] = [(item - reject_min) / (reject_max - reject_min + 1e-6) for item in reject_clip_score]
    
with open(target_path, mode='w', encoding='utf-8') as file:
    json.dump(mdpo_dataset, file, indent=4)

print('done.')