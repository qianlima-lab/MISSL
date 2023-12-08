

import torch
import torch.nn as nn
import torch.nn.functional as F


# head used for bert4rec
class DotProductPredictionHead(nn.Module):
    """share embedding parameters"""
    def __init__(self, d_model, num_items, token_embeddings):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.vocab_size = num_items + 1
        self.out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )
        self.ln = nn.LayerNorm(d_model)
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, x, b_seq, candidates=None):
        x = self.ln(self.out(x))  # B x H or M x H
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  # x : M x H
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
            emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
            logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
            logits += self.bias
        return logits