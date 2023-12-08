# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 20:15
# @Author  : Kyun_Ng
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import os
import torch
from torch import nn
import torch.nn.functional
import pytorch_lightning as pl
from .embedding import TypeAwareEmbedding, IntentPrototype
from .missl_layer import IntentEncoder, BiTransformerEncoder
from collections import Counter
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class MISSL(pl.LightningModule):
    def __init__(self,
                 max_len: int = 50, num_items: int = None, n_layer: int = None, n_head: int = None,
                 n_b: int = None, d_model: int = None, dropout: float = .0, buy: int = 2,
                 intent_num: int = 10, hhgt_num_head: int = 2, hhgt_num_layer: int = 1,
                 epsilon: float = 0.3, nb: int = 24, sim: str = 'cos', temp: float = 1.0,
                 item_beh_lambda: float = 0.05, next_item_lambda: float = 1.0, next_beh_lambda: float = 0.2,
                 inter_intent_lambda: float = 0.05, intra_intent_lambda: float = 0.05, weight: float = 0.6,
                 cl_sample_rate: float = 0.2):
        super().__init__()

        self.max_seq_length = max_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = self.hidden_size = d_model
        self.dropout_prob = dropout

        self.num_items = self.n_items = num_items
        self.n_b = self.n_beh = n_b

        self.n_intent = intent_num
        self.n_bs_intent = self.n_ba_intent = self.n_intent
        self.n_bs_intents = self.n_intent * self.n_beh
        self.n_ba_intents = self.n_intent
        self.n_intents = self.n_bs_intents + self.n_ba_intents
        self.epsilon = epsilon
        self.buy = buy
        self.nb = nb

        self.embedding = TypeAwareEmbedding(self.n_items+1, self.n_beh+1, self.hidden_size, self.max_seq_length, self.dropout_prob)
        self.intentPrototype = IntentPrototype(self.n_intents, self.hidden_size, self.dropout_prob)

        self.hhgt_num_head = hhgt_num_head
        self.hhgt_num_layer = hhgt_num_layer

        self.sim = sim
        self.temp = temp
        self.ib_lambda = item_beh_lambda
        self.ni_lambda = next_item_lambda
        self.nb_lambda = next_beh_lambda
        self.inter_intent_lambda = inter_intent_lambda
        self.intra_intent_lambda = intra_intent_lambda
        self.weight = weight
        self.cl_sample_rate = cl_sample_rate

        self.trm_encoder = BiTransformerEncoder(n_layers=self.n_layer, n_heads=self.n_head, n_b=self.n_b,
                                                hidden_size=self.hidden_size, inner_size=4*self.hidden_size, dropout=self.dropout_prob)

        self.intent_encoder = IntentEncoder(n_layers=self.hhgt_num_layer, n_heads=self.hhgt_num_head,
                                            n_b=self.n_beh, n_intent=self.n_intent, epsilon=self.epsilon,
                                            max_seq_length=self.max_seq_length, nb=self.nb, hidden_size=self.hidden_size,
                                            inner_size=4*self.hidden_size, dropout=self.dropout_prob)

    @staticmethod
    def get_attention_mask(item_seq, bidirectional=False):
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand((-1, -1, item_seq.size(-1), -1))  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask)
        return extended_attention_mask

    def get_attention_mask_item_intent(self, item_seq, type_seq):
        attention_mask = (item_seq > 0).long()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.repeat(1, 1, self.n_intents, 1).permute(2, 1, 0, 3)
        for beh_id in range(self.n_beh):
            attention_mask[beh_id*self.n_intent:(beh_id+1)*self.n_intent, 0] = (type_seq == beh_id+1).long()
        attention_mask = attention_mask.permute(2, 1, 0, 3)
        return attention_mask

    def get_type_cnt(self, type_seq):
        cnt = []
        for ts in type_seq:
            count = dict(Counter(filter(lambda x: 0 < x <= self.n_beh, ts.cpu().numpy())))
            for key in count:
                count[key] = max(1, int(count[key]))
            cnt.append(count)
        return cnt

    def forward(self, item_seq, type_seq):
        input_emb_item = self.embedding(item_seq=item_seq, mode='ips')
        input_emb_type = self.embedding(type_seq=type_seq, mode='tps')
        graph_input = self.embedding(item_seq=item_seq, type_seq=type_seq, mode='itp')
        type_cnt = self.get_type_cnt(type_seq)
        intent = self.intentPrototype()
        extended_attention_mask = self.get_attention_mask(item_seq)
        attention_mask1 = self.get_attention_mask_item_intent(item_seq, type_seq)  # [bs, 1, n_intents, maxlen]
        attention_mask2 = attention_mask1.permute(0, 1, 3, 2)  # [bs, 1, maxlen, n_intents]

        graph_item_output, graph_intent_output, idx_bs, idx_ba = self.intent_encoder(graph_input, intent, type_seq,
                                                                                        type_cnt, attention_mask1, attention_mask2)
        bs_intent, ba_intent = torch.split(graph_intent_output, (self.n_intent * self.n_beh, self.n_intent), dim=1)
        input_emb_item = (1.-self.weight)*input_emb_item + self.weight*graph_item_output
        item_emb_output, type_emb_output = self.trm_encoder(input_emb_item, input_emb_type, extended_attention_mask, type_seq)
        return item_emb_output, type_emb_output, graph_input, bs_intent, ba_intent, idx_bs, idx_ba