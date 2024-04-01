import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from train import get_dataset, get_model, get_or_build_tokenizer, run_validation, greedy_decode
from config import get_config, get_device, get_weights_file_path
from dataset import BilingualDataset

import evaluate
from tqdm import tqdm


class PerformanceEvaluation:
    sacrebleu_metric = evaluate.load("bleu")
    def __init__(self, config, device, preload):
        self.config = config
        self.device = device
        
        train_dataloader, validation_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
        model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)
        
        if preload:
            model_filename = get_weights_file_path(config, preload)
            print(f"Preloading model {model_filename}")
            state = torch.load(model_filename, map_location=device)
            model.load_state_dict(state["model_state_dict"])
        else:
            print("No model preloaded")
        
        self.model = model
        self.validation_dataloader = validation_dataloader
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        

    def run(self, samples):
        self.model.eval()
        references = []
        hypotheses = []
        
        count = 0
        with torch.no_grad():
            for batch in tqdm(self.validation_dataloader):
                inputs = batch["encoder_input"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                targets = batch["label"].to(self.device)
                
                model_out = greedy_decode(self.model, inputs, encoder_mask, self.source_tokenizer, self.config["seq_len"], self.device)
                model_out_text = self.target_tokenizer.decode(model_out.detach().cpu().numpy())
                
                target_text = self.target_tokenizer.decode(targets[0].detach().cpu().numpy())
                
                references.append([model_out_text])
                hypotheses.append(target_text)
                
                count += 1
                if count == samples:
                    break

        bleu_score = self.sacrebleu_metric.compute(predictions=hypotheses, references=references)
        return bleu_score
        

def main():
    parser = argparse.ArgumentParser(description="Performance evaluation for the trained model")
    parser.add_argument("--preload", type=str, help="File to preload the model from")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate")
    args = parser.parse_args()

    # Now you can access args.preload to determine if preload is True or False
    preload = args.preload
    samples = args.samples

    # Assuming you have other necessary parameters like config and device
    config = get_config()
    device = get_device()
    
    performance_eval = PerformanceEvaluation(config, device, preload)
    bleu_score = performance_eval.run(samples)
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"BLEU score: {bleu_score}")
    print(f"BLEU score: {bleu_score}")

if __name__ == "__main__":
    main()
