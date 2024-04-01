from datasets import load_dataset, load_metric
import numpy as np


def load_small_imdb():
    """Loads a small subset of the IMDB dataset, sliced into 3000 training examples,
    300 validation examples, and 300 test examples.
    """
    imdb = load_dataset("imdb")

    small_train_dataset = imdb["train"].shuffle(seed=42).select(list(range(3000)))
    small_val_dataset = imdb["train"].shuffle(seed=42).select(list(range(3000, 3300)))
    small_test_dataset = imdb["test"].shuffle(seed=42).select(list(range(300)))

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
