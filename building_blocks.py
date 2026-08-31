import torch


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.fan_in = fan_in
        self.weights = torch.randn(fan_in, fan_out)/fan_in**0.5
        self.biases = torch.zeros(fan_out) if bias else None

    def __call__(self, inp):
        prod = inp@self.weights
        if self.biases is not None:
            prod += self.biases
        self.out = prod
        return self.out

    def parameters(self):
        biases = [self.biases] if self.biases is not None else []
        return [self.weights] + biases

class Embedding:
    def __init__(self, vocab_size, d_emb):
        self.vocab_size = vocab_size
        self.d_emb = d_emb

        self.weight = torch.randn((vocab_size, d_emb))

    def __call__(self, inp):
        self.out = self.weight[inp]
        return self.out

    def parameters(self):
        return [self.weight]

    def reset_params(self):
        self.weight = torch.randn((self.vocab_size, self.d_emb))
        
class SelfAttention:
    def __init__(self, d_emb, heads, d_k = None, masked = False):
        self.heads = heads
        self.masked = masked
        self.d_v = d_k if d_k is not None else d_emb//heads
        self.Wq = [torch.randn(d_emb, self.d_v)/(d_emb**0.5) for _ in range(heads)]
        self.Wk = [torch.randn(d_emb, self.d_v)/(d_emb**0.5) for _ in range(heads)]
        self.Wv = [torch.randn(d_emb, self.d_v)/(d_emb**0.5) for _ in range(heads)]
        self.Wo = torch.randn(heads*self.d_v, d_emb)/((heads*self.d_v)**0.5)

    def __call__(self, X):
        out_lis = []
        for i in range(self.heads):
            Q, K = X@self.Wq[i], X@self.Wk[i]
            V = X@self.Wv[i]
            att_out = (Q @ (K.transpose(1, 2))) / ((self.d_v)**0.5)
            if self.masked:
                n = X.shape[1]
                inf = torch.ones(n, n)*float('-inf')
                inf = torch.stack([torch.tril(inf, diagonal = -1).T] * X.shape[0])
                att_out = torch.tril(att_out) + inf
            out_lis.append(torch.softmax(att_out, dim = 2) @ V)
        self.out = torch.cat(out_lis, dim = 2)
        if self.heads > 1:
            self.out = self.out@self.Wo

        return self.out
    def parameters(self):
        return self.Wq + self.Wk + self.Wv + [self.Wo]

class CrossAttention:
    def __init__(self, d_emb, heads, d_k = None):
        self.heads = heads
        self.d_v = d_k if d_k is not None else d_emb//heads
        self.Wq = [torch.randn(d_emb, self.d_v)/(d_emb**0.5) for _ in range(heads)]
        self.Wk = [torch.randn(d_emb, self.d_v)/(d_emb**0.5) for _ in range(heads)]
        self.Wv = [torch.randn(d_emb, self.d_v)/(d_emb**0.5) for _ in range(heads)]
        self.Wo = torch.randn(heads*self.d_v, d_emb)/((heads*self.d_v)**0.5)

    def __call__(self, Query, Key):
        out_lis = []
        for i in range(self.heads):
            Q, K = Query@self.Wq[i], Key@self.Wk[i]
            V = Key@self.Wv[i]
            att_out = (Q @ (K.transpose(1, 2))) / ((self.d_v)**0.5)
            out_lis.append(torch.softmax(att_out, dim = 2) @ V)
        self.out = torch.cat(out_lis, dim = 2)
        if self.heads > 1:
            self.out = self.out@self.Wo
        return self.out

    def parameters(self):
        return self.Wq + self.Wk + self.Wv + [self.Wo]

class PositionalEncode:
    def __init__(self, n_embd, max_len = 1024):
        # positional embeddings
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        C = -torch.log(torch.tensor(10000.0))
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (C / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe
    def __call__(self, X):
        self.out = X + torch.stack([self.pe[:X.size(1), :]] * X.size(0))
        return self.out

    def parameters(self):
        return []

class FFN:
    def __init__(self, d_emb, d_ff):
        self.W1 = torch.randn(d_emb, d_ff)*0.01
        self.b1 = torch.randn(1, d_ff)*0.001
        self.W2 = torch.randn(d_ff, d_emb)*0.01
        self.b2 = torch.randn(1, d_emb)*0.001

    def __call__(self, X):
        hidden_layer = X @ self.W1 + self.b1
        self.out = torch.max(torch.tensor(0), hidden_layer) @ self.W2 + self.b2
        return self.out

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
