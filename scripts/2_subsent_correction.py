import json
from tqdm import tqdm


mdpo_path = '/home/data/wyy/projects/SeVa/mdpo_llava_10k.json'
target_path = '/home/data/wyy/projects/SeVa/mdpo_llava_10k_reasoning.json'
    
    
with open(mdpo_path, mode='r', encoding='utf-8') as file:
    mdpo_dataset=json.load(file)
    

target_dataset = []

i = 0
for data in tqdm(mdpo_dataset):  
    # if len(data['chosen_split']) > 5 or len(data['reject_split']) > 20:
    if 'reasoning' in data['image_id']:
        target_dataset.append(data)
        i += 1
    
with open(target_path, mode='w', encoding='utf-8') as file:
    json.dump(target_dataset, file, indent=4)

print('done.')
print(i)