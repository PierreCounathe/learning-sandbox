import os

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import wandb

from p02_lora.load_model import load_pretrained_model
from p02_lora.utils import load_small_imdb, define_preprocess_function, compute_metrics
from p02_lora.config import get_config

def main():
    # Get the project's configuration
    config = get_config()

    # Log in to W&B and set the project
    wandb_config = config["wandb"]
    wandb.login(key=wandb_config["key"], host=wandb_config["host"])
    os.environ["WANDB_PROJECT"] = wandb_config["project"]
    os.environ["WANDB_LOG_MODEL"] = wandb_config["log_model"]

    # Load the model, tokenizer, prepare the datasets and data collator
    model, tokenizer = load_pretrained_model()

    imdb_datasets_dict = load_small_imdb()
    train_dataset = imdb_datasets_dict["train"]
    val_dataset = imdb_datasets_dict["val"]

    preprocess_function = define_preprocess_function(tokenizer)
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Freeze base model weights only (do not freeze pre_classifier and classifier)
    for param in model.base_model.parameters():
        param.requires_grad = False

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