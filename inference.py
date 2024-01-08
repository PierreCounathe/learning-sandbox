import torch
from torch.utils.data import Dataset, DataLoader
from train import get_dataset, get_model, get_or_build_tokenizer, run_validation
from config import get_config, get_weights_file_path
from dataset import BilingualDataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    
if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    config = get_config()
    config["batch_size"] = 1
    
    source_tokenizer = get_or_build_tokenizer(config, None, config["source_language"])
    target_tokenizer = get_or_build_tokenizer(config, None, config["target_language"])

    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
    
    while True:
        print("Enter an english sentence to translate ('exit' to quit):")
        sentence = input()
        if sentence == "exit":
            break
        validation_dataset_dictionary = {"translation": {"en": sentence, "fr": ""}}
        validation_input_dataset = CustomDataset([validation_dataset_dictionary])
        validation_dataset = BilingualDataset(validation_input_dataset, source_tokenizer, target_tokenizer, config["source_language"], config["target_language"], config["seq_len"])
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
        run_validation(model, validation_dataloader, source_tokenizer, target_tokenizer, config["seq_len"], device, lambda x: print(x), num_examples=1)
        