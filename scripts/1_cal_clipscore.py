import os
import re
import json
from transformers import CLIPModel, AutoProcessor
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse


def split_sentences(text):
    sentences = re.split(r'(?<=[^.])\.(?=\s*[A-Z])', text)
    cleaned_sentences = [sentence.strip() + '.' if not sentence.endswith('.') else sentence.strip() for sentence in sentences if sentence.strip()]
    
    return cleaned_sentences


class CustomDataset(Dataset):
    def __init__(self, dataset, image_dir, processor):
        self.dataset = dataset
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img_path = data['image_id']
        image = Image.open(os.path.join(self.image_dir, img_path))
        chosen = data['chosen_split']
        reject = data['reject_split']
        texts = chosen + reject
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True, truncation=True)
        return inputs, len(chosen)


def get_clip_scores(batch_inputs, model):
    batch_inputs['pixel_values'] = batch_inputs['pixel_values'].squeeze(0)
    batch_inputs['input_ids'] = batch_inputs['input_ids'].squeeze(0)
    batch_inputs['attention_mask'] = batch_inputs['attention_mask'].squeeze(0)
    with torch.no_grad():
        outputs = model(**batch_inputs)
    logits_per_image = outputs.logits_per_image[0].cpu().numpy()
    return logits_per_image.tolist()


def eval_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.data_json, 'r') as file:
        dataset = json.load(file)

    clip_model = CLIPModel.from_pretrained(args.clip_model_path).to(device)
    clip_processor = AutoProcessor.from_pretrained(args.clip_model_path)

    custom_dataset = CustomDataset(dataset, args.image_dir, clip_processor)
    dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=False)  # 调整batch_size根据你的显存大小

    for index, (batch_inputs, chosen_lengths) in tqdm(enumerate(dataloader)):
        batch_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
        results = get_clip_scores(batch_inputs, clip_model)
        
        dataset[index]['chosen_clip_score'] = results[:chosen_lengths[0]]
        dataset[index]['reject_clip_score'] = results[chosen_lengths[0]:]

    with open(args.output_path, 'w') as file:
        json.dump(dataset, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default='/home/data/wyy/projects/SeVa/llava_instruction_diffusion_step500_7b_iter_1_split_clip_score.json')
    parser.add_argument("--data_json", type=str, default='/home/data/wyy/projects/SeVa/llava_instruction_diffusion_step500_7b_iter_1_split.json')
    parser.add_argument("--image_dir", type=str, default='/home/data/wyy/datasets/VLFeedback/merged_images/')
    parser.add_argument("--clip_model_path", type=str, default='/home/data/wyy/checkpoints/clip-vit-large-patch14-336')
    args = parser.parse_args()

    eval_model(args)