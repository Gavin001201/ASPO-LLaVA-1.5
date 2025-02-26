import json
import random
from tqdm import tqdm


mdpo_path = '/home/data/wyy/projects/SeVa/mdpo_llava_10k.json'
sub_path1 = '/home/data/wyy/projects/SeVa/mdpo_llava_10k_detail.json'
sub_path2 = '/home/data/wyy/projects/SeVa/mdpo_llava_10k_conversation.json'
sub_path3 = '/home/data/wyy/projects/SeVa/mdpo_llava_10k_reasoning.json'
target_path = '/home/data/wyy/projects/SeVa/mdpo_llava_10k_resplit.json'
    
    
with open(mdpo_path, mode='r', encoding='utf-8') as file:
    mdpo_dataset=json.load(file)
  
with open(sub_path1, mode='r', encoding='utf-8') as file1:
    sub_dataset1=json.load(file1)
    
with open(sub_path2, mode='r', encoding='utf-8') as file2:
    sub_dataset2=json.load(file2)
    
with open(sub_path3, mode='r', encoding='utf-8') as file3:
    sub_dataset3=json.load(file3)  

resplit_dataset = sub_dataset1 + sub_dataset2 + sub_dataset3

random.shuffle(resplit_dataset)

img_list = []
for data in tqdm(resplit_dataset):  
    img_list.append(data['image_id'])
    
img_list= set(img_list)
    
with open(target_path, mode='w', encoding='utf-8') as file:
    json.dump(resplit_dataset, file, indent=4)

print('done.')