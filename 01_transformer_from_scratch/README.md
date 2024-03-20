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

<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28203%2C235%2C235%2C1%29&t=seti&wt=none&l=powershell&width=680&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=2x&wm=false&code=Enter%2520an%2520english%2520sentence%2520to%2520translate%2520%28%27exit%27%2520to%2520quit%29%253A%250AI%2520was%2520waiting%2520to%2520show%2520you%250A--------------------------------------------------------------------------------%250ASOURCE%253A%2520I%2520was%2520waiting%2520to%2520show%2520you%250ATARGET%253A%2520%250APREDICTED%253A%2520Je%2520t%2520%25E2%2580%2599%2520attendais%2520pour%2520te%2520montrer%2520.%250AEnter%2520an%2520english%2520sentence%2520to%2520translate%2520%28%27exit%27%2520to%2520quit%29%253A%250AIs%2520it%2520possible%253F%250A--------------------------------------------------------------------------------%250ASOURCE%253A%2520Is%2520it%2520possible%253F%250ATARGET%253A%2520%250APREDICTED%253A%2520Est%2520-%2520ce%2520possible%2520%253F%250AEnter%2520an%2520english%2520sentence%2520to%2520translate%2520%28%27exit%27%2520to%2520quit%29%253A%250AThis%2520model%2520does%2520not%2520know%2520what%2520a%2520GPU%2520is%21%250A--------------------------------------------------------------------------------%250ASOURCE%253A%2520This%2520model%2520does%2520not%2520know%2520what%2520a%2520GPU%2520is%21%250ATARGET%253A%2520%250APREDICTED%253A%2520Il%2520ne%2520sait%2520pas%2520ce%2520qu%2520%25E2%2580%2599%2520on%2520appelle%2520%21"
  style="width: 824px; height: 521px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>


## Train, run inference, evaluate

- **Train:** all parameters are defined in `config.py`, run `python3 train.py`
- **Inference:** run `python3 inference.py` to preload the model defined in `config.py` and run inferences in the terminal
- **Evaluate:** run `python3 performance_evaluation.py --preload 20 --samples 100` to run bleu score assessment on the validation set