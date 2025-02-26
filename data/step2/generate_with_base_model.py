import os
import re
import json
import tqdm
import torch
import argparse
import datetime
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import sys
sys.path.insert(0, "/home/data/wyy/projects/SeVa/seva")


from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


import torchvision.transforms as transforms
import random
from PIL import ImageFilter, ImageDraw

def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor




def add_diffusion_noise(image_tensor, noise_step=500):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd



def main(args):
    # Model
    disable_torch_init()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    model_name = get_model_name_from_path(args.model_path)
    if args.model_base is None:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path, 
            model_base=args.model_base, 
            model_name=model_name,
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base, 
            model_name="llava_lora_model",
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )

    conv_mode = "llava_v1"

    ## get question file
    with open(args.image_file_list, mode='r', encoding='utf-8') as fp:
        lines= json.load(fp)
    rank, word_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    step = len(lines) // word_size + 1
    start, end = rank * step, (rank + 1) * step
    results = []
    if int(os.environ["RANK"]) == 0:
        print("generating answers...")

    results = []
    for line in tqdm.tqdm(lines[start:end]):
        message_input = line["question"]

        image = line["image"].split('/')[-1]
        
        image_path_map = {'GQA':'GQA', 'VQA':'VQA/train2014', 'LVIS':'LVIS/train2017'}
        sub_path = line["image"].split('/')[0]
        
        if sub_path == 'GQA':
            sub_image_path = image_path_map['GQA']
        elif sub_path == 'COCO2015':
            sub_image_path = image_path_map['VQA']
        elif sub_path == 'LVIS':
            sub_image_path = image_path_map['LVIS']
        else:
            sub_image_path = line["image"].split('/')[0]
                
        image_path = os.path.join(args.image_path, sub_image_path, image)

        image = Image.open(image_path).convert("RGB")
        
        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        #############################
        # def visualize(matches, image_tensor, image_path, image_processor, model_config):
        #     image_processor.do_normalize = False
        #     image = Image.open(image_path).convert('RGB')
        #     image_tensor = process_images([image], image_processor, model_config)[0]

        #     # 移除批处理维度，并将通道顺序从 [C, H, W] 转换到 [H, W, C]
        #     tensor = image_tensor.squeeze(0).permute(1, 2, 0)

        #     # 将数据范围从 [0, 1] 转换到 [0, 255]，并转换为 uint8
        #     image = (tensor.numpy() * 255).astype(np.uint8)

        #     # 生成 PIL 图像
        #     image = Image.fromarray(image)

        #     width, height = image.size
        #     # 可视化定位区域, 绘制bounding box
        #     coordinates = []
        #     for item in matches:
        #         # 将每个字符串按逗号分割并转为浮点数
        #         coords = list(map(float, item.split(',')))
        #         coordinates.append(coords)
        #     for bbox in coordinates:
        #         draw = ImageDraw.Draw(image)
        #         x1 = width * bbox[0]
        #         y1 = height * bbox[1]
        #         x2 = width * bbox[2]
        #         y2 = height * bbox[3]
        #         draw.rectangle([x1, y1, x2, y2], outline="red", width=1)

        #     # 保存图像到本地
        #     target_path = "/home/data/wyy/projects/SeVa/data/custom_cot_dpo_train/VoCoT/visualize/" + '/'.join(image_path.split("/")[-2:])
        #     image.save(target_path)
    
        # pattern = r'\[\s*([0-9]*\.[0-9]+(?:,\s*[0-9]*\.[0-9]+)*)\s*\]'
        # matches = re.findall(pattern, line['chosen']) 
        # visualize(matches, image_tensor, image_path, image_processor, model.config)
        #############################

        # if len(args.augmentations) > 0 and "diffusion" in args.augmentations:
        #     image_tensor = add_diffusion_noise(image_tensor, args.noise_step)

        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = message_input

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '').replace('\n', '')
        results.append({
            "question": message_input,
            "answer": outputs,
            "image": line['image']
        })
        
    device = f"cuda:{torch.cuda.current_device()}"
    # convert dictionary -> tensor for gather all results in all ranks
    part_tensor = convert_dict_to_tensor(results, device)
    shape_tensor = torch.tensor(part_tensor.shape, device=device)
    shape_list = [shape_tensor.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
    torch.distributed.all_gather(shape_list, shape_tensor)

    # gather tensor
    max_shape = max(shape_list)
    part_tensor_pad = torch.zeros(max_shape).to(device)
    part_tensor_pad[:part_tensor.shape[0]] = part_tensor
    tensor_list = [part_tensor_pad.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
    torch.distributed.all_gather(tensor_list, part_tensor_pad)

    if int(os.environ["RANK"]) == 0:
        results_all_rank = []
        for tensor, shape in zip(tensor_list, shape_list):
            t = tensor.long()[:shape]
            _data = "".join([chr(t[i].item()) for i in range(t.shape[0])])
            _data = json.loads(_data)
            results_all_rank.extend(_data)
        # sort according to question_id
        # results_all_rank = sorted(results_all_rank, key=lambda x:x["question_id"])
        res_file = args.res_file
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, res_file), "w") as f:
            for res in results_all_rank:
                f.write(json.dumps(res)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--res_file", type=str, default="generate.jsonl")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    parser.add_argument('--augmentations', nargs='+', type=str, default=[])
    parser.add_argument("--noise_step", default=500, type=int)
    parser.add_argument("--image_file_list", default=None, type=str)
    
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["LOCAL_RANK"]),
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    
    args = parser.parse_args()
    main(args)