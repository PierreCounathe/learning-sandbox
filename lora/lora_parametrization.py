import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import DistilBertForSequenceClassification

class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1):
        super().__init__()

        self.lora_a = nn.Parameter(torch.zeros(rank, features_out))
        self.lora_b = nn.Parameter(torch.zeros(features_in, rank))
        nn.init.normal_(self.lora_a, mean=0, std=1)

        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            device = original_weights.device
            return original_weights + self.scale * (
                torch.matmul(self.lora_b, self.lora_a).view(original_weights.shape).to(device)
                )
        return original_weights

def linear_layer_parametrization(layer, rank=1, lora_alpha=1):
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(features_in, features_out, rank, lora_alpha)

def lora_parametrization(model, rank=1, lora_alpha=1):
    # Freeze base model weights only (do not freeze pre_classifier and classifier)
    for layer in model.distilbert.transformer.layer:
        parametrize.register_parametrization(
            layer.attention.q_lin, "weight", linear_layer_parametrization(
                layer.attention.q_lin, rank, lora_alpha
            )
        )
        parametrize.register_parametrization(
            layer.attention.k_lin, "weight", linear_layer_parametrization(
                layer.attention.k_lin, rank, lora_alpha
            )
        )
        parametrize.register_parametrization(
            layer.attention.v_lin, "weight", linear_layer_parametrization(
                layer.attention.v_lin, rank, lora_alpha
            )
        )

    for name, param in model.named_parameters():
        if "lora" not in name:
            print(f"Freezing original parameter {name}")
            param.requires_grad = False
        else:
            print(f"Unfreezing LoRA parameter {name}")
            param.requires_grad = True


class LoRADistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config=config)

    def apply_lora(self, rank=1, lora_alpha=1):
        lora_parametrization(self, rank, lora_alpha)

    def unfreeze_head_layers(self):
        for name, param in self.named_parameters():
            if "classifier" in name or "pre_classifier" in name:
                print(f"Unfreezing head parameter {name}")
                param.requires_grad = True
            else:
                print(f"Freezing base model parameter {name}")
                param.requires_grad = False