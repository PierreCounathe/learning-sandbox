from datasets import load_dataset, load_metric
import numpy as np

from lora.config import get_config


def load_small_imdb():
    """Loads a small subset of the IMDB dataset, sliced into N training examples,
    N/10 validation examples, and N/10 test examples.
    """
    dataset_size = get_config()["dataset_size"]
    
    imdb = load_dataset("imdb")

    small_train_dataset = imdb["train"].shuffle(seed=42).select(list(range(dataset_size)))
    small_val_dataset = imdb["train"].shuffle(seed=42).select(list(range(dataset_size, int(1.1*dataset_size))))
    small_test_dataset = imdb["test"].shuffle(seed=42).select(list(range(int(.1*dataset_size))))

    return {
        "train": small_train_dataset,
        "val": small_val_dataset,
        "test": small_test_dataset
    }

def define_preprocess_function(tokenizer):
    """Uses the tokenizer to define the function that processes examples by batch
    """
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    return preprocess_function

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score for a batch of predictions.
    """
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}
