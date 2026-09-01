# ShakespeareGPT

A GPT-style, character-level language model built from scratch in raw PyTorch tensors — no `nn.Module`, no `torch.optim`. Everything is hand-implemented. Trains on the shakespeare dataset(https://github.com/karpathy/char-rnn), as provided, to generate Shakespeare-flavored text one character at a time.

## Project structure

```
microgpt/
├── building_blocks.py   # Linear, Embedding, PositionalEncode, SelfAttention, CrossAttention, FFN
├── models.py             # EncoderLayer, DecoderLayer, DecoderOnlyModel
├── optim.py               # Adam, SGD, RMSprop, AdaGrad
├── data.py                 # vocab building, batching, optimizer-hyperparameter config
├── checkpoint.py          # save/load (model weights | architecture | vocab)
├── main.py                  # training entry point
├── generate.py             # text generation from a saved checkpoint
├── shakespeare.txt         # tinyshakespeare dataset
└── Notebooks               # Notebooks used while experimenting
      └── Transformer_implementation.ipynb
```

## Features

- Decoder-only transformer (causal self-attention, no cross-attention) built entirely from `torch.Tensor` operations
- Character-level tokenization, No external tokenizer used
- Four optimizers implemented from the underlying math: **Adam**, **SGD** (with momentum/dampening), **RMSprop**, **AdaGrad**
- warmup and decay schedule in loss as per the "Attention is all you need" paper
- Checkpointing — architecture and vocab are saved alongside weights, so `generate.py` can reload a model with no extra flags
- Can input values in command-line through `argparse`

## Installations required

```bash
pip install torch matplotlib
```

## Usage

### Train

```bash
python main.py
```

- This builds the vocabulary from `shakespeare.txt`,
- Trains a 4-layer decoder-only transformer by default, you can change that variable as mentioned below
- Periodically checkpoints to `checkpoint.pt`, and plots the train/val loss curve at the end.

Available flags:

| Flag | Default | Description |
|---|---|---|
| `--file-path` | `shakespeare.txt` | Path to training text |
| `--iterations` | `2000` | Number of training steps |
| `--batch-size` | `32` | Batch size |
| `--block-size` | `128` | Context length |
| `--d-emb` | `384` | Embedding dimension |
| `--heads` | `6` | Number of attention heads |
| `--n-decoder-layers` | `4` | Number of decoder layers |
| `--optimizer` | `adam` | One of `adam`, `sgd`, `rmsprop`, `adagrad` |
| `--warmup-steps` | `400` | LR warmup steps |
| `--save-path` | `checkpoints/checkpoint.pt` | Where to write the checkpoint |

Run `python main.py --help` for the full list, including per-optimizer hyperparameters for specific optimizer(`--lr`, `--beta1`, `--beta2`, `--momentum`, `--dampening`, `--gamma`, `--eps`).

Example, training with SGD and a larger context window:

```bash
python main.py --optimizer sgd --lr 0.01 --momentum 0.9 --block-size 256
```

Although, there isn't a way to shift to a different architecture transformer, I have coded a transformer with encoder too for experimentation and reference.

### Generate

```bash
python generate.py
```

Loads `checkpoint.pt` and autoregressively samples characters from a default prompt. Architecture and vocab are read directly from the checkpoint, so no model-size flags are needed here.

Usage example,

```bash
python generate.py --load-path checkpoint.pt --prompt "ROMEO:\n" --n-tokens 500
```

| Flag | Default | Description |
|---|---|---|
| `--load-path` | `checkpoints/checkpoint.pt` | Checkpoint to load |
| `--prompt` | `hamlet_default_prompt` | Seed text to continue from |
| `--n-tokens` | `1000` | Number of characters to generate |

hamlet_default_prompt :

```
HAMLET:
To be, or not to be, that is the question: whether 'tis 
nobler in the mind to suffer the slings and arrows of outrageous 
fortune, or to take arms against a sea of troubles.
```

## Acknowledgements and Goals
- It's a pretty small and slow model. Of course we could use more layers and more data and more of everything. But at the end of the day, is a practice project.
- I'd make it so that we can transfer the tensors to a different device.
- Tokenizers will also be added to the code.
