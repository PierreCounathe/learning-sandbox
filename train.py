import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import get_config, get_weights_file_path

from dataset import BilingualDataset
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item["translation"][language]


def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    dataset_raw = load_dataset("opus_books", f"{config['source_language']}-{config['target_language']}", split="train")
    
    source_tokenizer = get_or_build_tokenizer(config, dataset_raw, config["source_language"])
    target_tokenizer = get_or_build_tokenizer(config, dataset_raw, config["target_language"])
    
    train_split_size = int(0.9 * len(dataset_raw))
    validation_dataset_size = len(dataset_raw) - train_split_size
    train_dataset_raw, validation_dataset_raw = random_split(dataset_raw, [train_split_size, validation_dataset_size])
    
    train_dataset = BilingualDataset(train_dataset_raw, source_tokenizer, target_tokenizer, config["source_language"], config["target_language"], config["seq_len"])
    validation_dataset = BilingualDataset(validation_dataset_raw, source_tokenizer, target_tokenizer, config["source_language"], config["target_language"], config["seq_len"])
    
    max_source_len = 0
    max_target_len = 0
    
    for item in dataset_raw:
        source_ids = source_tokenizer.encode(item["translation"][config["source_language"]]).ids
        target_ids = target_tokenizer.encode(item["translation"][config["target_language"]]).ids
        max_source_len = max(max_source_len, len(source_ids))
        max_target_len = max(max_target_len, len(target_ids))
    
    print(f"Max length of source sentences:  {max_source_len}")
    print(f"Max length of target sentences:  {max_target_len}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    
    return train_dataloader, validation_dataloader, source_tokenizer, target_tokenizer


def get_model(config, source_vocab_size, target_vocab_size):
    model = build_transformer(source_vocab_size, target_vocab_size, config["seq_len"], config["seq_len"], config["d_model"])
    return model


def train_model(config):
    # Define the device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Training using device {device}")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, validation_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config["experiment_name"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id("[PAD]"), label_smoothing=.1).to(device)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (B, seq)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1 seq)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq, seq)
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)  # (B, seq, target_vocabulary_size)
            
            label = batch["label"].to(device)  # kj
            
            # (B, seq, target_vocabulary_size) -> (B * seq, target_vocabulary_size)
            loss = loss_fn(proj_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
            }, 
            model_filename
        )

if __name__ == "__main__":
    config = get_config()
    train_model(config)