from torch import nn as nn
import torch


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.apply(self._init_weights)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SegmentEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.se = nn.Embedding(2, d_model)
        self.apply(self._init_weights)

    def forward(self, x, idx):
        batch_size = x.size(0)
        return self.se.weight[idx].unsqueeze(0).repeat(batch_size, 1, 1)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TypeAwareEmbedding(nn.Module):
    """
        1. TokenEmbedding : normal embedding matrix
        2. PositionEmbedding: position embedding matrix
        3. BehaviorEmbedding: behavior embedding matrix
    """

    def __init__(self, vocab_size, type_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.typeToken = nn.Embedding(num_embeddings=type_size, embedding_dim=embed_size, padding_idx=0)
        self.segment_embedding = SegmentEmbedding(d_model=embed_size)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.apply(self._init_weights)

    def forward(self, item_seq=None, type_seq=None, mode='itp'):
        if mode == 'itp':
            assert item_seq is not None
            assert type_seq is not None
            x = self.token(item_seq) + self.typeToken(type_seq) + self.position(item_seq)
        elif mode == 'ips':
            assert item_seq is not None
            x = self.token(item_seq) + self.position(item_seq) + self.segment_embedding(item_seq, 0)
        elif mode == 'tps':
            assert type_seq is not None
            x = self.typeToken(type_seq) + self.position(type_seq) + self.segment_embedding(type_seq, 1)
        elif mode == 'ip':
            assert item_seq is not None
            # print(item_seq)
            x = self.token(item_seq) + self.position(item_seq)
        elif mode == 'ps':
            assert item_seq is not None
            x = self.position(item_seq) + self.segment_embedding(item_seq, 0)
        elif mode == 'tp':
            assert type_seq is not None
            x = self.typeToken(type_seq) + self.position(type_seq)
        elif mode == 's':
            assert item_seq is not None
            x = self.segment_embedding(item_seq, 0)
        elif mode == 'i':
            x = self.token(item_seq)
        else:
            raise NotImplementedError
        return self.dropout(x)

    @staticmethod
    def _init_weights(module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class IntentPrototype(nn.Module):
    def __init__(self, n_intents, embed_size, dropout=0.1):
        super().__init__()

        self.intent = nn.Parameter(torch.rand(n_intents, embed_size))
        self.dropout = nn.Dropout(p=dropout)
        self.intent.data.normal_(mean=0.0, std=0.02)

    def forward(self):
        return self.dropout(self.intent.unsqueeze(0))