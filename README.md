# microgpt

A GPT-style, character-level language model built from scratch in raw PyTorch tensors — no `nn.Module`, no `torch.optim`. Every layer (linear, embedding, multi-head attention, layer norm, feed-forward) and every optimizer (Adam, SGD, RMSprop, AdaGrad) is hand-implemented. Trains on the [tinyshakespeare](https://github.com/karpathy/char-rnn) dataset to generate Shakespeare-flavored text one character at a time.

## Features

- Decoder-only transformer (causal self-attention, no cross-attention) built entirely from `torch.Tensor` operations
- Character-level tokenization — no subword vocabulary, no external tokenizer
- Four optimizers implemented from the underlying math: **Adam**, **SGD** (with momentum/dampening), **RMSprop**, **AdaGrad**
- Transformer-style learning rate warmup + decay schedule
- Checkpointing — architecture and vocab are saved alongside weights, so `generate.py` can reload a model with no extra flags
- Fully CLI-configurable via `argparse`

## Project structure

```
microgpt/
├── building_blocks.py   # Linear, Embedding, PositionalEncode, SelfAttention, CrossAttention, FFN
├── models.py             # EncoderLayer, DecoderLayer, DecoderOnlyModel (the model actually trained)
├── optim.py               # Adam, SGD, RMSprop, AdaGrad -- implemented from scratch
├── data.py                 # vocab building, batching, optimizer-hyperparameter config
├── checkpoint.py          # save/load model weights + architecture + vocab
├── main.py                  # training entry point
├── generate.py             # text generation from a saved checkpoint
└── shakespeare.txt         # tinyshakespeare dataset
```

## Installation

```bash
pip install torch matplotlib
```

## Usage

### Train

```bash
python main.py
```

This builds the vocabulary from `shakespeare.txt`, trains a 4-layer decoder-only transformer, periodically checkpoints to `checkpoint.pt`, and plots the train/val loss curve at the end.

Common flags:

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
| `--save-path` | `checkpoint.pt` | Where to write the checkpoint |

Run `python main.py --help` for the full list, including per-optimizer hyperparameters (`--lr`, `--beta1`, `--beta2`, `--momentum`, `--dampening`, `--gamma`, `--eps`).

Example, training with SGD and a larger context window:

```bash
python main.py --optimizer sgd --lr 0.01 --momentum 0.9 --block-size 256
```

### Generate

```bash
python generate.py
```

Loads `checkpoint.pt` and autoregressively samples characters from a default Hamlet prompt. Architecture and vocab are read directly from the checkpoint, so no model-size flags are needed here.

```bash
python generate.py --load-path checkpoint.pt --prompt "ROMEO:\n" --n-tokens 500
```

| Flag | Default | Description |
|---|---|---|
| `--load-path` | `checkpoint.pt` | Checkpoint to load |
| `--prompt` | *(Hamlet soliloquy)* | Seed text to continue from |
| `--n-tokens` | `1000` | Number of characters to generate |

## Notes

This project prioritizes understanding the mechanics of a transformer and its optimizers over speed — attention loops over heads in Python rather than batching them, and there's no GPU-specific optimization. It's meant as a from-scratch learning exercise, not a production training script.
