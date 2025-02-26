import json
import spacy
# 加载英文模型
nlp = spacy.load("en_core_web_sm")
from tqdm import tqdm


def split_sentences_with_spacy(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


mdpo_path = '/home/data/wyy/projects/SeVa/llava_instruction_diffusion_step500_7b_iter_1.jsonl'
target_path = '/home/data/wyy/projects/SeVa/llava_instruction_diffusion_step500_7b_iter_1_split.json'
    
    
with open(mdpo_path, mode='r', encoding='utf-8') as file:
    mdpo_dataset=json.load(file)
    

target_dataset = []

for data in tqdm(mdpo_dataset):  
    new_data = {}
    new_data['chosen'] = data['chosen']
    new_data['reject'] = data['reject']
    new_data['question'] = data['question']
    # new_data['question'] = data['prompt'].split('USER: ')[1].replace('<image>', '').replace('ASSISTANT:', '').strip()
    new_data['image_id'] = data['image_id']
    new_data['chosen_split'] = split_sentences_with_spacy(data['chosen'])
    new_data['reject_split'] = split_sentences_with_spacy(data['reject'])
    target_dataset.append(new_data)
    
    
with open(target_path, mode='w', encoding='utf-8') as file:
    json.dump(target_dataset, file, indent=4)

print('done.')