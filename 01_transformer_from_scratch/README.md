# Transformer From Scratch

The goal here it to build an encoder-decoder transformer, as described in the original Attention is all you need paper, for machine translation.

## Results

The model was trained on a subset (first 20k rows) of the english - french [opus books](https://huggingface.co/datasets/opus_books/viewer/en-fr) dataset. It was trained for roughly two hours on a V100 GPU. The configuration that was used is stored in `config.py`. For reference, the GPU was not overloaded, a slightly higher batch_size might lead to faster training on a similar GPU.

We compare the bleu score of the model (on 100 validation inferences) after 5, 10, 15 and 20 epochs of training.

| # epochs      | BLEU score |
| ----------- | ----------- |
| 5      | 0.04       |
| 10   | 0.12        |
| 15   | 0.28        |
| 20   | 0.58        |

More training, on the full dataset, and a larger architecture (more blocks) would probably lead to significantly better results. However, the goal is not to build a perfect translator, but to build a system that's capable of learning. This system is capable of learning, as demonstrated in the table above.

We also provide examples of the fully trained model below one example that lies in the training set, one that does not, and one complex example with unknown tokens (`'model'`, `'GPU'`).


<blockquote class="imgur-embed-pub" lang="en" data-id="rTocmap"  ><a href="//imgur.com/rTocmap">carbon</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## Train, run inference, evaluate

- **Train:** all parameters are defined in `config.py`, run `python3 train.py`
- **Inference:** run `python3 inference.py` to preload the model defined in `config.py` and run inferences in the terminal
- **Evaluate:** run `python3 performance_evaluation.py --preload 20 --samples 100` to run bleu score assessment on the validation set