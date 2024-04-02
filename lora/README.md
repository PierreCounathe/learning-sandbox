# LoRA From Scratch


## Results

## Reproduce experiment
1. Clone the repo, create and setup a virtual environment
```shell
git clone https://github.com/PierreCounathe/simple-implementations
cd simple-implementations
python -m venv .env
source .env/bin/activate
pip install -r lora/requirements.txt
```

2. Setup your Weights and Biases api key (can be found at https://wandb.ai/authorize once an account has been created)

In a notebook:
```python
%env WAND_API_KEY=my_key
```
In a shell:
```shell
export WANDB_API_KEY=my_key
```

3. If using Google Colab or Colab's terminal, make sure to add the cloned repo to `PYTHONPATH`

In a notebook:
```python
import sys
sys.append("content/simple-implementations")
```
In a shell:
```shell
export PYTHONPATH=$PYTHONPATH"/content/simple-implementations
```


