import torch
import torch.nn.functional as F

from building_blocks import *
from data import batch


class EncoderLayer:
    def __init__(self, d_emb, d_ff, multi_head):
        self.self_attention = SelfAttention(d_emb, multi_head)
        self.feed_forward = FFN(d_emb, d_ff)
        self.layer_norm_gamma1 = torch.ones(1, d_emb)
        self.layer_norm_beta1 = torch.zeros(1, d_emb)
        self.layer_norm_gamma2 = torch.ones(1, d_emb)
        self.layer_norm_beta2 = torch.zeros(1, d_emb)
        self.eps = 1e-5

        self.sub_layers = [self.self_attention, self.feed_forward]
        self.params = [self.layer_norm_gamma1, self.layer_norm_beta1, self.layer_norm_gamma2, self.layer_norm_beta2]

    def __call__(self, X_pos_encoded):
        X_s_attended = self.self_attention(X_pos_encoded)
        X_add1 = X_pos_encoded + X_s_attended
        X_norm1 = self.layer_norm_gamma1 * ((X_add1 - X_add1.mean(dim = -1).unsqueeze(-1)) / (X_add1.std(dim = -1).unsqueeze(-1)**2 + self.eps)**0.5) + self.layer_norm_beta1

        X_ffn = self.feed_forward(X_norm1)
        X_add2 = X_norm1 + X_ffn
        X_norm2 = self.layer_norm_gamma2 * ((X_add2 - X_add2.mean(dim = -1).unsqueeze(-1)) / (X_add2.std(dim = -1).unsqueeze(-1)**2 + self.eps)**0.5) + self.layer_norm_beta2
        self.out = X_norm2

        return self.out

    def parameters(self):
        params = list(self.params)
        for layer in self.sub_layers:
            params += layer.parameters()
        return params

class DecoderLayer:
    def __init__(self, d_emb, d_ff, multi_head, cross_attention = True):
        self.self_attention = SelfAttention(d_emb, multi_head, masked = True)
        self.cross_attention = CrossAttention(d_emb, multi_head)
        self.ca = cross_attention
        self.feed_forward = FFN(d_emb, d_ff)
        self.layer_norm_gamma1 = torch.ones(1, d_emb)
        self.layer_norm_beta1 = torch.zeros(1, d_emb)
        self.layer_norm_gamma2 = torch.ones(1, d_emb)
        self.layer_norm_beta2 = torch.zeros(1, d_emb)
        self.layer_norm_gamma3 = torch.ones(1, d_emb)
        self.layer_norm_beta3 = torch.zeros(1, d_emb)
        self.eps = 1e-5

        self.sub_layers = [self.self_attention, self.feed_forward]
        self.params = [self.layer_norm_gamma1, self.layer_norm_beta1, self.layer_norm_gamma3, self.layer_norm_beta3]

        if cross_attention:
            self.params += [self.layer_norm_gamma2, self.layer_norm_beta2]
            self.sub_layers.append(self.cross_attention)

    def __call__(self, X_pos_encoded, Xk = None):
        X_s_attended = self.self_attention(X_pos_encoded)
        X_add1 = X_pos_encoded + X_s_attended
        X_norm1 = self.layer_norm_gamma1 * ((X_add1 - X_add1.mean(dim = -1).unsqueeze(-1)) / (X_add1.std(dim = -1).unsqueeze(-1)**2 + self.eps)**0.5) + self.layer_norm_beta1
        X_next = X_norm1

        if self.ca:
            X_c_attended = self.cross_attention(X_next, Xk)
            X_add2 = X_next + X_c_attended
            X_norm2 = self.layer_norm_gamma2 * ((X_add2 - X_add2.mean(dim = -1).unsqueeze(-1)) / (X_add2.std(dim = -1).unsqueeze(-1)**2 + self.eps)**0.5) + self.layer_norm_beta2
            X_next = X_norm2

        X_ffn = self.feed_forward(X_next)
        X_add3 = X_next + X_ffn
        X_norm3 = self.layer_norm_gamma3 * ((X_add3 - X_add3.mean(dim = -1).unsqueeze(-1)) / (X_add3.std(dim = -1).unsqueeze(-1)**2 + self.eps)**0.5) + self.layer_norm_beta3
        self.out = X_norm3

        return self.out

    def parameters(self):
        params = list(self.params)
        for layer in self.sub_layers:
            params += layer.parameters()
        return params

# kept for reference / testing
class Transformer:
    def __init__(self, encoders, decoders):
        self.encoders = encoders  # list containing encoder objects
        self.decoders = decoders  # list containing decoder objects
        self.params = [p for layer in (self.encoders + self.decoders) for p in layer.parameters()]

    def __call__(self, input_encoding, output_encoding):
        x = input_encoding
        for encoder in self.encoders:
            x = encoder(x)
        # x will be the encoding by now
        y = output_encoding
        for decoder in self.decoders:
            y = decoder(x, y)
        return y

    def parameters(self):
        return self.params
    
class DecoderOnlyModel:
    def __init__(self, vocab_size, d_emb, heads, n_decoder_layers, d_ff = 0):
        self.d_emb = d_emb
        self.heads = heads
        self.n_decoder_layers = n_decoder_layers
        self.vocab_size = vocab_size
        self.d_ff = 4*d_emb if d_ff <= 0 else d_ff

        self.Embed = Embedding(vocab_size, d_emb)
        self.Pos_enc = PositionalEncode(d_emb)
        
        self.Decoders = [DecoderLayer(d_emb, self.d_ff, heads, cross_attention=False) for _ in range(n_decoder_layers)]
        
        self.lin = Linear(d_emb, vocab_size)
        self.lin.weights.data *= 0.01
        self.layers = [self.Embed, self.Pos_enc] + self.Decoders + [self.lin] 
        self.params = [p for layer in self.layers for p in layer.parameters()]
        
        for p in self.params:
            p.requires_grad = True

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        self.out = X
        return self.out
    
    def parameters(self):
        return self.params

    def train(self, optim, warmup_steps, iterations, eval_interval, train_data, val_data, batch_size, block_size, save_path, save_data):
        t_loss, v_loss = [], []
        optim.describe()
        
        save_data["block_size"] = block_size
        del save_data["train_data"]
        del save_data["val_data"]
        
        for step in range(iterations):
            X, Y = batch(split = "train", train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size)
            out = self(X)
            loss = F.cross_entropy(out.transpose(1, 2), Y)
    
            optim.zero_grad(set_to_none = True)
            loss.backward()
    
            update_steps = step + 1
            lr = (self.d_emb ** -0.5) * min(update_steps ** -0.5, update_steps * warmup_steps ** -1.5)
            optim.lr = lr
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            optim.step()
            
            if step % eval_interval == 0:
                with torch.no_grad():
                    X_v, Y_v = batch("val", train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size)
                    out_v = self(X_v)
                    loss_v = F.cross_entropy(out_v.transpose(1, 2), Y_v)
                
                t_loss.append(loss.item())
                v_loss.append(loss_v.item())
    
                if step % 100 == 0:
                    from checkpoint import save_checkpoint
                    save_checkpoint(model=self, data=save_data, path=save_path)
                    print(f"iteration: {step:5d} | loss: {loss.item():.4f} | val loss: {loss_v.item():.4f}")
                    print(f"----learn: {lr:.10f}")
                    
        return t_loss, v_loss