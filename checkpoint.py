import torch

from models import DecoderOnlyModel


def save_checkpoint(model,data,path):
    torch.save(
        {
            "params": [p.detach().clone() for p in model.parameters()],
            "vocab_size": model.vocab_size,
            "d_emb": model.d_emb,
            "d_ff": model.d_ff,
            "heads": model.heads,
            "n_decoder_layers": model.n_decoder_layers,
            "data": data
        },
        path,
    )
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path):
    checkpoint = torch.load(path, weights_only=False)

    model = DecoderOnlyModel(
        checkpoint["vocab_size"],
        checkpoint["d_emb"],
        checkpoint["d_ff"],
        checkpoint["heads"],
        checkpoint["n_decoder_layers"],
    )

    params = model.parameters()
    saved = checkpoint["params"]
    if len(params) != len(saved):
        raise ValueError(
            f"Checkpoint has {len(saved)} parameter tensors but the current config"
            f"produces {len(params)}. Did d_emb/d_ff/heads/n_decoder_layers in config "
            "change since this checkpoint was saved?"
        )
    for p, s in zip(params, saved):
        p.data.copy_(s)
        p.requires_grad = True

    data = checkpoint["data"]
    
    return model, data
