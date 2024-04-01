import os

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import torch
from torch import nn
from torch.nn.utils import parametrize
import wandb

from lora.load_model import load_pretrained_model
from lora.utils import load_small_imdb, define_preprocess_function, compute_metrics
from lora.config import get_config


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

def main():
    # Get the project's configuration
    config = get_config()

    # Log in to W&B and set the project
    wandb_config = config["wandb"]
    wandb.login(key=wandb_config["key"], host=wandb_config["host"])
    os.environ["WANDB_PROJECT"] = wandb_config["project"]
    os.environ["WANDB_LOG_MODEL"] = wandb_config["log_model"]

    # Load the model and tokenizer
    model, tokenizer = load_pretrained_model()

    # Add LoRA weights, and freeze the other parameters
    lora_parametrization(model)

    # Prepare the datasets and data collator
    imdb_datasets_dict = load_small_imdb()
    train_dataset = imdb_datasets_dict["train"]
    val_dataset = imdb_datasets_dict["val"]

    preprocess_function = define_preprocess_function(tokenizer)
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define the training arguments and the trainer
    training_args = TrainingArguments(
        **config["training_arguments"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Run a first evaluation before fine-tuning
    trainer.evaluate()

    # Fine-tune the model
    trainer.train()

    # Run a final evaluation after fine-tuning
    trainer.evaluate()

if __name__ == "__main__":
    main()
