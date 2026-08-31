import argparse

import torch

from checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny Char-level LM on Shakespeare.")
    parser.add_argument("--load-path", type=str, default="checkpoints/checkpoint.pt")
    parser.add_argument("--n-tokens", type=int, default=1000)
    parser.add_argument("--prompt", type=str, default="""HAMLET:
To be, or not to be, that is the question: whether 'tis 
nobler in the mind to suffer the slings and arrows of outrageous 
fortune, or to take arms against a sea of troubles.
""")

    return parser.parse_args()

def generate(prompt, n_tokens, load_path):
    generated_txt = prompt
    
    model, data = load_checkpoint(load_path)
    stoi = data["stoi"]
    block_size = data["block_size"]
    itos = data["itos"]
    encode = lambda txt : [stoi[t] for t in txt]

    X = torch.tensor(encode(prompt))[-block_size:].view( 1, -1) if len(prompt) >= block_size else torch.tensor(encode(prompt)).view( 1, -1)
    for step in range(n_tokens):
        with torch.no_grad():
            out = model(X)
            relevent = out[:, -1, :][0]

            t = torch.softmax(relevent, dim = 0)

            idx = torch.multinomial(t, num_samples = 1).item()
        ch = itos[idx]

        generated_txt += ch
        print(generated_txt)
        print("________________________")
        if len(X) < block_size:
            X = torch.tensor([X.tolist()[0] + [idx]])
        else:
            X = torch.tensor([X.tolist()[0][1:] + [idx]])
    return generated_txt

if __name__ == "__main__":
    args = parse_args()
    generate(prompt=args.prompt, n_tokens=args.n_tokens, load_path=args.load_path)
