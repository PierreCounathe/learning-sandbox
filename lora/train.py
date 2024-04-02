import argparse
import os

import torch
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import wandb

from lora.load_model import load_pretrained_model
from lora.utils import load_small_imdb, define_preprocess_function, compute_metrics
from lora.config import get_config


def main(lora_rank=None, lora_alpha=None, load_model=None, save_model=None, fine_tune="head"):
    # Get the project's configuration
    config = get_config()

    # Log in to W&B and set the project
    wandb_config = config["wandb"]
    wandb.login(key=wandb_config["key"], host=wandb_config["host"])
    os.environ["WANDB_PROJECT"] = wandb_config["project"]
    os.environ["WANDB_LOG_MODEL"] = wandb_config["log_model"]

    # Load the model and tokenizer
    model, tokenizer = load_pretrained_model(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        load_model=load_model,
        save_model=save_model,
        fine_tune=fine_tune)

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
    
    if save_model:
        print(f"Saving model to {save_model}")
        torch.save(model.state_dict(), save_model)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for your program")

    parser.add_argument("--rank", type=int, help="Specify rank (optional)")
    parser.add_argument("--alpha", type=float, help="Specify alpha (optional)")
    parser.add_argument("--load_model", type=str, help="Specify path to load model (optional)")
    parser.add_argument("--save_model", type=str, help="Specify path to save model (optional)")
    parser.add_argument("--fine_tune", type=str, help="Specify whether to fine-tune the head or the projections (optional)")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(lora_rank=args.rank, lora_alpha=args.alpha, load_model=args.load_model, save_model=args.save_model, fine_tune=args.fine_tune)

