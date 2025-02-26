import torch
from typing import Optional
from peft import PeftConfig, PeftModel
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
import sys
sys.path.append("/home/data/wyy/projects/SeVa/seva")
from llava.model import *

@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_name: Optional[str] = field(default="/home/data/wyy/projects/SeVa/checkpoints/bz_10-lr_2e-6-lora_r_1024-scaling_factor_1-new_data-loglikelihood-7b", metadata={"help": "the adapter name"})
    base_model_name: Optional[str] = field(default="/home/data/wyy/checkpoints/llava-v1.5-7b", metadata={"help": "the base model name"})
    output_name: Optional[str] = field(default="/home/data/wyy/projects/SeVa/checkpoints/merged_new_data2_iter_1", metadata={"help": "the merged model name"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.output_name is not None, "please provide the output name of the merged model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
if peft_config.task_type == "SEQ_CLS":
    # The sequence classification task is used for the reward model in PPO
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = LlavaLlamaForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")