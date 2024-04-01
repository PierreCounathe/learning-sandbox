import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import get_config, get_device, get_weights_file_path

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def greedy_decode(model, source, source_mask, source_tokenizer, max_len, device):
    sos_idx = source_tokenizer.token_to_id("[SOS]")
    eos_idx = source_tokenizer.token_to_id("[EOS]")
    
    # Precompte the decoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # Build a mask for the target (decoder_input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Get the next token, by using linear and softmax on the last predicted embedding
        probabilities = model.project(out[:, -1])
        _, next_word = torch.max(probabilities, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

    
def run_validation(model, validation_dataset, source_tokenizer, target_tokenizer, max_len, device, print_message, num_examples=2):
    model.eval()
    count = 0
    
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, source_tokenizer, max_len, device)
            
            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = target_tokenizer.decode(model_out.detach().cpu().numpy())
            
            print_message("-"*console_width)
            print_message(f"SOURCE: {source_text}")
            print_message(f"TARGET: {target_text}")
            print_message(f"PREDICTED: {model_out_text}")
            
            if count == num_examples:
                break
            

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
    dataset_raw = load_dataset("opus_books", f"{config['source_language']}-{config['target_language']}", split="train[:20000]")
    
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
    device = get_device()
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
        state = torch.load(model_filename, map_location=device)
        initial_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id("[PAD]"), label_smoothing=.1).to(device)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        model.train()
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
            
        run_validation(model, validation_dataloader, source_tokenizer, target_tokenizer, config["seq_len"], device, lambda msg: batch_iterator.write(msg))
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