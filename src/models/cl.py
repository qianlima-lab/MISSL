# -*- coding: utf-8 -*-
# @Time    : 2023/5/22 16:23
# @Author  : Kyun_Ng
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import torch
import torch.nn.functional as F
from torch import nn

nce_fct = nn.CrossEntropyLoss()


def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        # mask[i, :batch_size] = 0
        # mask[batch_size + i, batch_size:] = 0
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def info_nce_item_beh(z_i, z_j, temp, batch_size, _sim='cos'):
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)
    if _sim == 'cos':
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp  # [256, 256]
    elif _sim == 'dot':
        sim = torch.mm(z, z.T) / temp
    else:
        raise NotImplementedError('sim must in [cos, dot]')

    # logits[mask] = -10000.
    # info_nce = self.nce_fct(logits, labels)
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(batch_size)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    info_nce = nce_fct(logits, labels)
    return info_nce


def info_nce_inter_intent(item, bs_i, ba_i, idx_bs, idx_ba, mask, i_t, temp=1.0, sim='cos',
                          n_beh=4, n_intent=10, n_bs_intents=40, n_ba_intents=10):
    """
    item: [bs, maxlen, dim]
    bs_i: [bs, n_intent*self.n_beh, dim]
    ba_i: [bs, n_intent, dim]
    idx_bs: [bs, maxlen, 1]
    idx_ba: [bs, maxlen, 1]
    mask: [bs, maxlen]
    i_t: [bs, maxlen]
    """
    bs, maxlen, dim = item.size()
    i_t = i_t.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, n_intent)
    mask_bs_ii = torch.eq(idx_bs, idx_bs.permute(0, 2, 1))
    mask_ba_ii = torch.eq(idx_ba, idx_ba.permute(0, 2, 1))
    bs_mask = torch.eye(n_bs_intents, dtype=torch.bool, device=item.device).unsqueeze(0).expand(bs, -1, -1)
    ba_mask = torch.eye(n_ba_intents, dtype=torch.bool, device=item.device).unsqueeze(0).expand(bs, -1, -1)

    idx_bs_ji = F.one_hot(idx_bs.squeeze(-1), num_classes=n_bs_intents)
    idx_ba_ji = F.one_hot(idx_ba.squeeze(-1), num_classes=n_ba_intents)
    idx_bs_ji[~mask.unsqueeze(2).expand(-1, -1, n_bs_intents)] = 0
    idx_ba_ji[~mask.unsqueeze(2).expand(-1, -1, n_ba_intents)] = 0
    idx_bs_ji = idx_bs_ji.permute(0, 2, 1).contiguous().view(-1, maxlen)
    idx_ba_ji = idx_ba_ji.permute(0, 2, 1).contiguous().view(-1, maxlen)
    idx_bs_pad = torch.zeros((bs * n_bs_intents, n_bs_intents), device=item.device)
    idx_ba_pad = torch.zeros((bs * n_ba_intents, n_ba_intents), device=item.device)
    idx_bs_ji = torch.cat((idx_bs_ji, idx_bs_pad), dim=-1)
    idx_ba_ji = torch.cat((idx_ba_ji, idx_ba_pad), dim=-1)

    idx_bs = idx_bs % n_beh
    idx_bs = idx_bs.view(-1, 1)[mask.view(-1, 1)]
    idx_ba = idx_ba.view(-1, 1)[mask.view(-1, 1)]

    if sim == 'cos':
        item = torch.nn.functional.normalize(item, dim=-1)
        bs_i = torch.nn.functional.normalize(bs_i, dim=-1)
        ba_i = torch.nn.functional.normalize(ba_i, dim=-1)

    score_ii = torch.einsum('bij,bkj->bik', item, item) / temp  # [bs, maxlen, maxlen]
    bs_score_ij = torch.einsum('bij,bkj->bik', item, bs_i) / temp  # [bs, maxlen, n_intents]
    bs_score_ji = bs_score_ij.permute(0, 2, 1)  # [bs, n_bs_intents, maxlen]
    bs_score_jj = torch.einsum('bij,bkj->bik', bs_i, bs_i) / temp  # [bs, n_bs_intents, n_bs_intents]
    ba_score_ij = torch.einsum('bij,bkj->bik', item, ba_i) / temp  # [bs, maxlen, n_ba_intents]
    ba_score_ji = ba_score_ij.permute(0, 2, 1)  # [bs, n_ba_intents, maxlen]
    ba_score_jj = torch.einsum('bij,bkj->bik', ba_i, ba_i) / temp  # [bs, n_ba_intents, n_ba_intents]

    score_ii = score_ii.masked_fill(~mask.unsqueeze(1).expand(-1, maxlen, -1), -1e30)
    score_ii = score_ii.masked_fill(mask_bs_ii, -1e30)
    score_ii = score_ii.masked_fill(mask_ba_ii, -1e30)
    bs_score_ji = bs_score_ji.masked_fill(~mask.unsqueeze(1).expand(-1, n_bs_intents, -1), -1e30)
    ba_score_ji = ba_score_ji.masked_fill(~mask.unsqueeze(1).expand(-1, n_ba_intents, -1), -1e30)
    bs_score_jj = bs_score_jj.masked_fill(bs_mask, -1e30)
    ba_score_jj = ba_score_jj.masked_fill(ba_mask, -1e30)

    score_ii = score_ii.view(-1, maxlen)[mask.view(-1, 1).expand(-1, maxlen)].view(-1, maxlen)  # [bs*maxlen', maxlen]
    bs_score_ij = torch.gather(bs_score_ij.view(bs, maxlen, n_beh, n_intent), 2,
                               (i_t + n_beh - 1) % n_beh).squeeze(2).view(-1, n_intent)[mask.view(-1, 1).expand(-1, n_intent)].view(-1, n_intent)  # [bs*maxlen', n_intents]
    bs_score_ji = bs_score_ji.contiguous().view(-1, maxlen)  # [bs*n_bs_intents, maxlen]
    bs_score_jj = bs_score_jj.view(-1, n_bs_intents)  # [bs*n_bs_intents, n_bs_intents]
    ba_score_ij = ba_score_ij.view(-1, n_ba_intents)[mask.view(-1, 1).expand(-1, n_intent)].view(-1, n_intent)  # [bs*maxlen', n_ba_intents]
    ba_score_ji = ba_score_ji.contiguous().view(-1, maxlen)  # [bs*n_bs_intents, maxlen]
    ba_score_jj = ba_score_jj.view(-1, n_ba_intents)  # [bs*n_bs_intents, n_ba_intents]

    bs_ij = torch.cat((bs_score_ij, score_ii), dim=-1)  # [bs*maxlen, n_intent], [bs*maxlen, maxlen]
    bs_ji = torch.cat((bs_score_ji, bs_score_jj), dim=-1)  # [bs*n_bs_intents, maxlen], [bs*n_bs_intents, n_bs_intents]
    ba_ij = torch.cat((ba_score_ij, score_ii), dim=-1)  # [bs*maxlen, n_intent], [bs*maxlen, maxlen]
    ba_ji = torch.cat((ba_score_ji, ba_score_jj), dim=-1)  # [bs*n_bs_intents, maxlen], [bs*n_bs_intents, n_bs_intents]
    # print(idx_bs[~mask.view(-1)])
    # print(bs_ij.size(), idx_bs.size(), mask.size())
    info_nce_ij = nce_fct(bs_ij, idx_bs) + nce_fct(ba_ij, idx_ba)
    info_nce_ji = nce_fct(bs_ji, idx_bs_ji) + nce_fct(ba_ji, idx_ba_ji)
    # print('!!!!!!!!!!!!!!!!!!!!!!!', info_nce_ij + info_nce_ji, '!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return info_nce_ij + info_nce_ji


def info_nce_intra_intent(bs_i, ba_i, idx_bs, idx_ba, mask, temp=1.0, sim='cos',
                          n_intents=50, n_bs_intents=40, n_ba_intents=10):
    """
    bs_i: [bs, n_bs_intents, dim]
    ba_i: [bs, n_ba_intents, dim]
    idx_bs: [bs, maxlen, 1]
    idx_ba: [bs, maxlen, 1]
    """
    bs = bs_i.size(0)
    bs_p = torch.arange(bs, device=bs_i.device) * n_bs_intents
    ba_p = torch.arange(bs, device=bs_i.device) * n_ba_intents
    bs_p = bs_p.unsqueeze(1).unsqueeze(2)
    ba_p = ba_p.unsqueeze(1).unsqueeze(2)

    idx_bs_p = (idx_bs + bs_p).view(-1, 1)[mask.view(-1, 1)]
    idx_ba_r = idx_ba.view(-1, 1)[mask.view(-1, 1)]
    idx_ij = torch.zeros((bs * n_bs_intents, n_intents), device=bs_i.device)
    idx_ij[idx_bs_p, idx_ba_r] = 1

    idx_ba_p = (idx_ba + ba_p).view(-1, 1)[mask.view(-1, 1)]
    idx_bs_r = idx_bs.view(-1, 1)[mask.view(-1, 1)]
    idx_ji = torch.zeros((bs * n_ba_intents, n_intents), device=bs_i.device)
    idx_ji[idx_ba_p, idx_bs_r] = 1

    if sim == 'cos':
        bs_i = F.normalize(bs_i, dim=-1)
        ba_i = F.normalize(ba_i, dim=-1)

    sim_ij = torch.einsum('bij,bkj->bik', bs_i, ba_i) / temp  # [bs, n_bs_intents, n_ba_intents]
    sim_ji = sim_ij.permute(0, 2, 1)  # [bs, n_ba_intents, n_bs_intents]
    sim_ii = torch.einsum('bij,bkj->bik', bs_i, bs_i) / temp  # [bs, n_bs_intents, n_bs_intents]
    sim_jj = torch.einsum('bij,bkj->bik', ba_i, ba_i) / temp  # [bs, n_ba_intents, n_ba_intents]

    bs_mask = torch.eye(n_bs_intents, dtype=torch.bool, device=bs_i.device).unsqueeze(0).expand(bs, -1, -1)
    ba_mask = torch.eye(n_ba_intents, dtype=torch.bool, device=bs_i.device).unsqueeze(0).expand(bs, -1, -1)
    if sim_ii.dtype == torch.float16:
        sim_ii = sim_ii.masked_fill(bs_mask == 1, -65500)
        sim_jj = sim_jj.masked_fill(ba_mask == 1, -65500)
    else:
        sim_ii = sim_ii.masked_fill(bs_mask == 1, -1e30)
        sim_jj = sim_jj.masked_fill(ba_mask == 1, -1e30)

    sim_ij = sim_ij.view(-1, n_ba_intents)
    sim_ji = sim_ji.contiguous().view(-1, n_bs_intents)
    sim_ii = sim_ii.view(-1, n_bs_intents)
    sim_jj = sim_jj.view(-1, n_ba_intents)

    sim_ij = torch.cat((sim_ij, sim_ii), dim=-1)  # [bs*n_bs_intents, n_ba_intents], [bs*n_bs_intents, n_bs_intents]
    sim_ji = torch.cat((sim_ji, sim_jj), dim=-1)  # [bs*n_ba_intents, n_bs_intents], [bs*n_ba_intents, n_ba_intents]
    # print(nce_fct(sim_ij, idx_ij), nce_fct(sim_ji, idx_ji))
    info_nce = nce_fct(sim_ij, idx_ij) + nce_fct(sim_ji, idx_ji)

    return info_nce