from typing import Any
import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset_raw, source_tokenizer, target_tokenizer, source_language, target_language, seq_len) -> None:
        super().__init__()
        
        self.dataset_raw = dataset_raw
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([self.source_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.source_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.source_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self):
        return len(self.dataset_raw)
    
    def __getitem__(self, index) -> Any:
        source_target_pair = self.dataset_raw[index]
        source_text = source_target_pair["translation"][self.source_language]
        target_text = source_target_pair["translation"][self.target_language]
        
        encoder_input_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_text).ids
        
        encoder_padding_needed = self.seq_len - len(encoder_input_tokens) - 2
        decoder_padding_needed = self.seq_len - len(decoder_input_tokens) - 1
        
        if encoder_padding_needed < 0 or decoder_padding_needed < 0:
            raise ValueError("Sentence is too long")
        
        # "[SOS]" + source sentence + "[EOS]" + padding
        encoder_input = torch.concat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_padding_needed, dtype=torch.int64)
            ]
        )
        
        # "[SOS]" + target sentence + padding
        decoder_input = torch.concat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_padding_needed, dtype=torch.int64)
            ]
        )
        
        # target_sentence + "[EOS]" + padding
        # Given "[SOS]" the decoder should find the first word
        # Given the full sentence, the decoder should find that following token is "[EOS]"
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_padding_needed, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, 1, seq) & (1, seq, seq)
            "label": label,
            "source_text": source_text,
            "target_text": target_text
        }
        
def causal_mask(size: int):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0