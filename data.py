import torch

def make_data(train_ratio, path):
    with open(path, "r") as f:
        text = f.read()
    alphabet = sorted(set(text))
    stoi = {letter : i for i, letter in enumerate(alphabet)}
    itos = {i : letter for i, letter in enumerate(alphabet)}
    vocab_size = len(alphabet)
    encode = lambda txt : [stoi[t] for t in txt]

    encoded_text = torch.tensor(encode(text)).long()

    n = int(train_ratio*len(encoded_text))
    train_data = encoded_text[:n]
    val_data = encoded_text[n:]
    
    return {"vocab_size" : vocab_size,
            "train_data" : train_data,
            "val_data" : val_data,
            "stoi" : stoi,
            "itos" : itos
            }
    
def batch(split, batch_size, train_data, val_data, block_size):
    split_dict = {
        "train": train_data,
        "val": val_data,
    }
    data = split_dict[split]
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

def build_config(args):
    optim_defaults = {
        "adam" : {
            "lr" : 0.1, 
            "beta1" : 0.999,
            "beta2" : 0.999,
            "eps" : 1e-8
        },
        "rmsprop" : {
            "lr" : 0.01,
            "gamma" : 0.9,
            "eps" : 1e-5
        },
        "adagrad" : {
            "lr" : 0.01, 
            "eps" : 1e-5
        },
        "sgd" : {
            "lr" : 0.001, 
            "momentum" : 0.0,
            "dampening" : 0.0
        }
    }

    defaults = optim_defaults[args.optimizer]
    for key, default_val in defaults.items():
        user_val = getattr(args, key, None)
        defaults[key] = user_val if user_val is not None else default_val

    return optim_defaults

def print_state(state):
    for k, v in state.items():
        print(f"              {k:10s} : {v}")