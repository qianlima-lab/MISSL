# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 0:50
# @Author  : Kyun_Ng
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LN(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(LN, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        return self.norm(x1 + self.dropout(x2))


class BiTransformerEncoder(nn.Module):
    def __init__(self, n_layers=2, n_heads=2, n_b=4, hidden_size=64, inner_size=256, dropout=0.1):
        super().__init__()
        self.layer = nn.ModuleList([BiTransformerBlock(n_heads, n_b, hidden_size, inner_size, dropout) for _ in range(n_layers)])

    def forward(self, hidden_states1, hidden_states2, mask, b_seq):
        for layer_module in self.layer:
            hidden_states1, hidden_states2 = layer_module(hidden_states1, hidden_states2, mask, b_seq)
        return hidden_states1, hidden_states2


class BiTransformerBlock(nn.Module):
    def __init__(self, attn_heads, n_b, hidden, feed_forward_hidden, dropout,):
        super().__init__()
        self.biAttention = BiAttention(n_b=n_b, h=attn_heads, d_model=hidden, dropout=dropout)
        self.iPff = BehaviorSpecificPFF(hidden, feed_forward_hidden, n_b, True, dropout)
        self.tPff = BehaviorSpecificPFF(hidden, feed_forward_hidden, n_b, True, dropout)
        self.norm = nn.ModuleList([LN(hidden, dropout) for _ in range(4)])

    def forward(self, x1, x2, mask, b_seq):
        _x1, _x2 = self.biAttention(x1, x2, mask, b_seq)
        x1, x2 = self.norm[0](x1, _x1), self.norm[1](x2, _x2)
        x1, x2 = self.norm[2](x1, self.iPff(x1, b_seq)), self.norm[3](x2, self.tPff(x2, b_seq))
        return x1, x2


class BiAttention(nn.Module):
    def __init__(self, n_b, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.n_b = n_b
        self.d_k = d_model // h
        self.h = h

        self.linear_layers_item = nn.Parameter(torch.randn(3, self.n_b+1, d_model, self.h, self.d_k))
        self.linear_layers_item.data.normal_(mean=0.0, std=0.02)

        self.linear_layers_type = nn.Parameter(torch.randn(3, self.n_b+1, d_model, self.h, self.d_k))
        self.linear_layers_type.data.normal_(mean=0.0, std=0.02)

        self.attention = Attention(dropout)

    def forward(self, hidden1, hidden2, mask, b_seq):
        batch_size, seq_len = hidden1.size(0), hidden1.size(1)
        iQuery, iKey, iValue = [torch.einsum("bnd, Bdhk, bnB->bhnk", x, self.linear_layers_item[lli], F.one_hot(b_seq, num_classes=self.n_b+1).float())
                                for lli, x in zip(range(3), (hidden1, hidden1, hidden1))]

        tQuery, tKey, tValue = [torch.einsum("bnd, Bdhk, bnB->bhnk", x, self.linear_layers_type[lli], F.one_hot(b_seq, num_classes=self.n_b+1).float())
                                for lli, x in zip(range(3), (hidden2, hidden2, hidden2))]
    
        hidden1 = self.attention(tQuery, iKey, iValue, mask=mask)
        hidden1 = hidden1.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    
        hidden2 = self.attention(iQuery, tKey, tValue, mask=mask)
        hidden2 = hidden2.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    
        return hidden1, hidden2


class BehaviorSpecificPFF(nn.Module):
    def __init__(self, d_model, d_ff, n_b, bpff=False, dropout=0.1):
        super().__init__()
        self.n_b = n_b
        self.bpff = bpff
        if bpff and n_b > 1:
            self.pff = nn.ModuleList(
                [PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout) for _ in range(n_b)])
        else:
            self.pff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def multi_behavior_pff(self, x, b_seq):
        outputs = [torch.zeros_like(x)]
        for i in range(self.n_b):
            outputs.append(self.pff[i](x))
        return torch.einsum('nBTh, BTn -> BTh', torch.stack(outputs, dim=0),
                            F.one_hot(b_seq, num_classes=self.n_b + 1).float())

    def forward(self, x, b_seq=None):
        if self.bpff and self.n_b > 1:
            return self.multi_behavior_pff(x, b_seq)
        else:
            return self.pff(x)


class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def make_noise(scores):
        noise = torch.rand(scores.shape).cuda()
        noise = -torch.log(-torch.log(noise))
        return scores + noise

    def forward(self, query, key, value, mask=None, topk=None):
        #fixme visualization
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            if len(mask.shape) == 2:
                mask = (mask[:, :, None] & mask[:, None, :]).unsqueeze(1)
            if scores.dtype == torch.float16:
                scores = scores.masked_fill(mask == 0, -65500)
            else:
                scores = scores.masked_fill(mask == 0, -1e30)
        if topk is not None:
            p_attn = self.dropout(nn.functional.softmax(scores, dim=-1))
            # p_attn = nn.functional.softmax(scores, dim=-1)
            p_attn = topk(p_attn)
            if isinstance(p_attn, list):
                return torch.matmul(p_attn[0], value), p_attn[1], p_attn[2]
        else:
            p_attn = self.dropout(nn.functional.softmax(scores, dim=-1))
        return torch.matmul(p_attn, value)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.apply(self._init_weights)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class IntentEncoder(nn.Module):
    def __init__(self, n_layers=2, n_heads=2, n_b=4, n_intent=10, epsilon=1,
                 max_seq_length=50, nb=20, hidden_size=64, inner_size=256, dropout=0.1):
        super().__init__()
        self.n_l = n_layers
        self.layer = IntentBlock(n_heads, n_b, n_intent, epsilon, max_seq_length, nb, hidden_size, inner_size, dropout)
        self.attn1 = nn.Parameter(torch.randn(1, n_layers+1, hidden_size))
        self.attn2 = nn.Parameter(torch.randn(1, n_layers+1, hidden_size))
        self.attn1.data.normal_(mean=0.0, std=0.02)
        self.attn2.data.normal_(mean=0.0, std=0.02)

    def forward(self, item, intent, b_seq, type_cnt, mask1, mask2):
        items, intents = [item], [intent.expand(item.size(0), -1, -1)]
        idx_bs = idx_ba = None
        for _ in range(self.n_l):
            item, intent, idx_bs, idx_ba = self.layer(item, intent, b_seq, type_cnt, mask1, mask2)
            items.append(item)
            intents.append(intent)
        items = torch.stack(items, dim=0).permute(1, 0, 2, 3)
        intents = torch.stack(intents, dim=0).permute(1, 0, 2, 3)
        s1 = nn.functional.softmax(torch.einsum('bkld, bkd-> bkl', items, self.attn1), dim=1)
        s2 = nn.functional.softmax(torch.einsum('bkld, bkd-> bkl', intents, self.attn2), dim=1)
        item = torch.einsum('bkl, bkld-> bld', s1, items)
        intent = torch.einsum('bkl, bkld-> bld', s2, intents)
        return item, intent, idx_bs, idx_ba


class IntentBlock(nn.Module):
    def __init__(self, n_h=2, n_b=4, n_i=10, epsilon=1, max_seq_length=50,
                 nb=20, hidden_size=64, inner_size=256, dropout=0.1):
        super().__init__()
        self.n_b = n_b
        self.n_i = n_i
        self.intentExtract = intentExtractor(n_h=n_h, n_i=n_i, n_b=n_b, d_model=hidden_size, dropout=dropout, nb=nb, max_seq_len=max_seq_length)
        self.intentFusion = intentFusionor(n_h=n_h, n_i=n_i, n_b=n_b, d_model=hidden_size, dropout=dropout, epsilon=epsilon)
        self.hgnn0 = HGNN(n_b=n_b, n_i=n_i)
        self.hgnn1 = HGNN(n_b=n_b, n_i=n_i)
        self.intentPFF = BehaviorSpecificPFF(d_model=hidden_size, d_ff=inner_size, n_b=n_b, bpff=True, dropout=dropout)
        self.itemPFF = BehaviorSpecificPFF(d_model=hidden_size, d_ff=inner_size, n_b=n_b, bpff=True, dropout=dropout)
        self.norm = nn.ModuleList([LN(hidden_size, dropout) for _ in range(4)])

    def forward(self, item, intent, b_seq, type_cnt, mask1, mask2):
        intent_b_seq = torch.LongTensor([i for i in range(self.n_b+1) for _ in range(self.n_i)]).unsqueeze(0).expand(item.size(0), -1).to(item.device)
        _intent = self.intentExtract(item, intent, mask1, b_seq, intent_b_seq, type_cnt)
        intent = self.norm[0](intent, _intent)
        _intent = self.intentPFF(intent, intent_b_seq)
        intent = self.norm[1](intent, _intent)
        intent = self.hgnn1(self.hgnn0(intent, self.n_i), self.n_i)
        _item, sIdx, aIdx = self.intentFusion(item, intent, mask2, b_seq, intent_b_seq)
        item = self.norm[2](item, _item)
        _item = self.itemPFF(item, b_seq)
        item = self.norm[3](item, _item)
        return item, intent, sIdx, aIdx


class intentExtractor(nn.Module):
    def __init__(self, n_h, n_i, n_b, d_model, dropout, nb, max_seq_len):
        super().__init__()
        self.n_h = n_h
        self.n_i = n_i
        self.n_b = n_b
        self.n_is = self.n_i*(self.n_b+1)
        self.d_k = d_model // n_h
        self.nb = nb
        self.max_seq_len = max_seq_len

        self.linear_layers_item = nn.Parameter(torch.randn(2, self.n_b + 2, d_model, self.n_h, self.d_k))
        self.linear_layers_intent = nn.Parameter(torch.randn(1, self.n_b + 1, d_model, self.n_h, self.d_k))
        self.linear_layers_item.data.normal_(mean=0.0, std=0.02)
        self.linear_layers_intent.data.normal_(mean=0.0, std=0.02)
        self.attention = Attention(dropout)

    @staticmethod
    def top_k(input_tensor, k):
        score_des, idx1 = torch.sort(input_tensor, dim=-1, descending=True)
        idx1_sort, idx2 = torch.sort(idx1, dim=-1)
        top_k_mask = idx2 >= k
        return top_k_mask

    @staticmethod
    def get_cn(x, nb=24, n=50):
        if 0 <= x < nb / 4:
            return x
        f1 = nb / 4 + int(np.log(4 * x / nb) / np.log(4 * n / nb) * nb / 4)
        upper_bound = nb / 2 - 1
        return int((f1 + upper_bound - np.abs(f1 - upper_bound)) / 2)

    def get_k(self, attention_score, type_cnt):
        bs, _, n_i, _ = attention_score.size()
        res = torch.zeros((bs, 1, n_i, 1), device=attention_score.device)
        if n_i == self.n_i:
            for bid, tc in enumerate(type_cnt):
                res[bid, 0] = self.get_cn(sum(tc.values()), nb=self.nb, n=self.max_seq_len)
        else:
            for bid, tc in enumerate(type_cnt):
                for hid in range(self.n_b):
                    tc_ = tc.get(hid+1, 0)
                    res[bid, 0, hid * self.n_i:(hid + 1) * self.n_i] = self.get_cn(tc_, nb=self.nb, n=self.max_seq_len)
        res = res.expand(-1, self.n_h, -1, -1)
        return res

    def intent_extract_top_k(self, scores, type_cnt):
        k = self.get_k(scores, type_cnt)
        mask = self.top_k(scores, k)
        scores = scores.masked_fill(mask, 0.0)  # todo abla
        return scores

    def forward(self, item, intent, mask, b_seq, b_seq2, type_cnt):
        bs, maxlen, dim = item.size()
        intent = intent.expand(bs, -1, -1)
        mKeyBs, mValueBs = [torch.einsum("bnd, Bdhk, bnB->bhnk", x, self.linear_layers_item[l], F.one_hot(b_seq, num_classes=self.n_b+2).float())
                            for l, x in zip(range(2), (item, item))]
        mKeyBa, mValueBa = [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_item[l, -1]) for l, x in zip(range(2), (item, item))]
        tQuery = torch.einsum("bnd, Bdhk, bnB->bhnk", intent, self.linear_layers_intent[0], F.one_hot(b_seq2, num_classes=self.n_b+1).float())
        tQueryBs, tQueryBa = torch.split(tQuery, (self.n_b*self.n_i, self.n_i), dim=2)
        xBs = self.attention(tQueryBs, mKeyBs, mValueBs, mask[:, :, :self.n_b*self.n_i, :], lambda x: self.intent_extract_top_k(x, type_cnt))
        xBa = self.attention(tQueryBa, mKeyBa, mValueBa, mask[:, :, -self.n_i:, :], lambda x: self.intent_extract_top_k(x, type_cnt))
        x = torch.concat((xBs, xBa), dim=2)
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.n_h * self.d_k)
        return x


class intentFusionor(nn.Module):
    def __init__(self, n_h, n_i, n_b, d_model, dropout, epsilon):
        super().__init__()
        self.h = n_h
        self.n_i = n_i
        self.n_b = n_b
        self.d_k = d_model // n_h
        self.k = int(self.n_i * epsilon)
        self.attention = Attention(dropout)

        self.linear_layers_item = nn.Parameter(torch.randn(1, self.n_b + 1, d_model, self.h, self.d_k))
        self.linear_layers_intent = nn.Parameter(torch.randn(2, self.n_b + 1, d_model, self.h, self.d_k))
        self.linear_layers_item.data.normal_(mean=0.0, std=0.02)
        self.linear_layers_intent.data.normal_(mean=0.0, std=0.02)

    @staticmethod
    def top_k(input_tensor, k):
        score_des, idx1 = torch.sort(input_tensor, dim=-1, descending=True)
        idx1_sort, idx2 = torch.sort(idx1, dim=-1)
        top_k_mask = idx2 >= k
        return top_k_mask

    def intent_fusion_top_k(self, scores):
        bs, n_heads, maxlen, _ = scores.size()

        bs_topk_mask = self.top_k(scores[:, :, :, :-self.n_i], self.k)
        ba_topk_mask = self.top_k(scores[:, :, :, -self.n_i:], self.k)
        topk_mask = torch.cat([bs_topk_mask, ba_topk_mask], dim=-1)
        bs_belong = self.top_k(scores[:, :, :, :-self.n_i], 1)
        ba_belong = self.top_k(scores[:, :, :, -self.n_i:], 1)

        bs_belong = (torch.where(~bs_belong[:, 0, :, :])[-1]).view(bs, maxlen, -1)
        ba_belong = (torch.where(~ba_belong[:, 0, :, :])[-1]).view(bs, maxlen, -1)
        scores = scores.masked_fill(topk_mask, 0.0)  # todo abla

        return [scores, bs_belong, ba_belong]

    def forward(self, item, intent, mask, b_seq, b_seq2):
        bs, maxlen, dim = item.size()
        tKey, tValue = [torch.einsum("bnd, Bdhk, bnB->bhnk", x, self.linear_layers_intent[l], F.one_hot(b_seq2, num_classes=self.n_b + 1).float())
                        for l, x in zip(range(2), (intent, intent))]
        mQuery = torch.einsum("bnd, Bdhk, bnB->bhnk", item, self.linear_layers_item[0], F.one_hot(b_seq, num_classes=self.n_b + 1).float())
        x, sIdx, aIdx = self.attention(mQuery, tKey, tValue, mask, lambda x: self.intent_fusion_top_k(x))
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return x, sIdx, aIdx


class HGNN(nn.Module):
    def __init__(self, n_b, n_i, dropout=0.1):
        super().__init__()
        self.h = nn.Parameter(torch.randn(n_b+1, n_i, n_i))
        self.h.data.normal_(mean=0.0, std=0.02)
        self.activation = nn.ReLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, x, n_i=10):
        x = x.view(x.size(0), -1, n_i, x.size(-1))
        _x = torch.einsum('BmK, bBKd->bBmd', self.h, x)
        return self.activation((x+self.dp(_x)).view(x.size(0), -1, x.size(-1)))