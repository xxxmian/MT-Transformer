import torch
import torch.nn as nn
from layer import EncoderLayer, DecoderLayer
from embed import Embedder, PositionalEncoder
from sublayer import Norm
import copy
from config import hidden_size, n_layers, heads, dropout, device


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, hidden_size)
        self.pe = PositionalEncoder(hidden_size, dropout=dropout)
        self.layers = get_clones(EncoderLayer(hidden_size, heads, dropout), N)
        self.norm = Norm(hidden_size)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def get_model(src_vocab_size, trg_vocab_size):
    assert dropout < 1

    model = Transformer(src_vocab_size, trg_vocab_size, hidden_size, n_layers, heads, dropout)

    """if load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    """

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


    model = model.to(device)

    return model

