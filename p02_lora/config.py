import os
import torch

def get_config():
    """Returns the project's configuration json
    """
    return {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "training_arguments": {
            "output_dir": "lora-from-scratch",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "num_train_epochs": 5,
            "weight_decay": 0.01,
            "save_strategy": "epoch",
            "report_to": ["wandb"]
        },
        "wandb": {
            "project": "lora-from-scratch",
            "log_model": "checkpoint",
            "host": "https://api.wandb.ai",
            "key": os.environ["WANDB_API_KEY"]
        }
    }

def get_device():
    """Returns the available device, in order of priority: cuda, mps, cpu
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")
    return device
