# LoRA From Scratch

In this folder, I compare the efficiency of fine-tuning a Transformer-based classifier's head vs. fine-tuning its Q, K and V projection matrices on a binary classification task. The fine-tuned model is [Distilbert-Base-Uncased fine-tuned on SST2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), and the dataset used is [Stanford's IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb).

LoRA is defined in `lora_parametrization.py`:
```python
class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1):
        super().__init__()

        self.lora_a = nn.Parameter(torch.zeros(rank, features_out))
        self.lora_b = nn.Parameter(torch.zeros(features_in, rank))
        nn.init.normal_(self.lora_a, mean=0, std=1)

        self.scale = alpha / rank
        self.enabled = True
```

## Results

[This report](https://wandb.ai/pierrecounathe/lora-from-scratch/reports/LoRA-vs-Head-fine-tuning--Vmlldzo3Mzc5NDky) contains the W&B logs of the four main runs.

Dataset size (train, val)| Training Configuration | Training Time | Validation F1
| ----------- | ----------- | ----------- | ----------- |
| N/A, 1k | Raw model from HF   | N/A       | 89.05%
| 10k, 1k | HEAD 15 epochs   | 15 minutes       | 90.98%
| 10k, 1k | LoRA (rank=1) 15 epochs    | 30 minutes | 91.29%
| 10k, 1k | LoRA (rank=16) 15 epochs   | 30 minutes | **91.63%**
| 10k, 1k | LoRA (rank=16) 10 epochs + HEAD 5 epochs | 27 minutes | 91.54%

- All training has been conducted on Colab using a V100 GPU, using only 10k samples from the dataset.
- The HEAD of the model has about 600k parameters. LoRA with rank=1 has about 45k parameters, and LoRA with rank=16 has about 700k parameters.
- Fine-tuning the head only is considerably faster than fine-tuning projection matrices with LoRA. This is due to the fact that when fine-tuning the classifier (and pre-classifier) only, the backpropagation's computational graph only includes those two layers as gradient don't have to flow any further. For the same reason, the GPU RAM used is only 3GB, vs. 10GB when fine-tuning with LoRA.
- However, fine-tuning the projection layers with LoRA leads to better performance (almost 1.5 times better than HEAD fine-tuning: almost 3% f1 increase vs. 2% f1 increase).
- The fact that `LoRA (rank=16) 15 epochs` leads to better results than `LoRA (rank=16) 10 epochs + HEAD 5 epochs` (although statistical significance is uncertain), leads to think that the information ingested in the model via head fine-tuning is included in the information ingested in the model via LoRA.


## Reproduce experiment
Clone the repo, create and setup a virtual environment
```shell
git clone https://github.com/PierreCounathe/simple-implementations
cd simple-implementations
python -m venv .env
source .env/bin/activate
pip install -r lora/requirements.txt
```

Use of the `train.py` script:
- HEAD fine-tuning `python lora/train.py --fine_tune head`
- LoRA fine-tuning `python lora/train.py --rank 16 --fine_tune lora --save_model path.pt`
- Use LoRA fine-tuned model and continue with HEAD fine-tuning `python lora/train.py --rank 16 --fine_tune head --load_model path.pt`


## Setup/Appendix
1. Setup your Weights and Biases api key (can be found at https://wandb.ai/authorize once an account has been created)

In a notebook:
```python
%env WAND_API_KEY=my_key
```
In a shell:
```shell
export WANDB_API_KEY=my_key
```

2. If using Google Colab or Colab's terminal, make sure to add the cloned repo to `PYTHONPATH`

In a notebook:
```python
import sys
sys.append("content/simple-implementations")
```
In a shell:
```shell
export PYTHONPATH=$PYTHONPATH:"/content/simple-implementations"
```


