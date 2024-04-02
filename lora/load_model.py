from transformers import AutoTokenizer
import torch

from lora.config import get_device, get_config
from lora.lora_parametrization import LoRADistilBertForSequenceClassification

CONFIG = get_config()

def load_pretrained_model(model_name=CONFIG["model_name"], lora_rank=None, lora_alpha=None, load_model=None, save_model=None, fine_tune="head"):
    """Loads and returns a model for sequence classification and its associated tokenizer.
    """
    device = get_device()
    model = LoRADistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    if lora_rank or lora_alpha:
        lora_rank = lora_rank or 1
        lora_alpha = lora_alpha or 1
        print(f"Applying LoRA with rank {lora_rank} and alpha {lora_alpha}")
        model.apply_lora(rank=lora_rank, lora_alpha=lora_alpha)
    
    model.freeze_all_layers()
    if fine_tune == "head":
        print("Unfreezing head layers")
        model.unfreeze_head_layers()
    elif fine_tune == "lora":
        print("Unfreezing LoRA layers")
        model.unfreeze_lora_layers()
    if load_model:
        print(f"Loading saved model from {load_model}")
        model.load_state_dict(torch.load(load_model))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
