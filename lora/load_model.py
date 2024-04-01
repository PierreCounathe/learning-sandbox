from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lora.config import get_device, get_config

CONFIG = get_config()

def load_pretrained_model(model_name=CONFIG["model_name"]):
    """Loads and returns a model for sequence classification and its associated tokenizer.
    """
    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "This was a great movie"
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    output = model(**encoded_input)
    print("input: ", text)
    print("output: ", output)
    return model, tokenizer
