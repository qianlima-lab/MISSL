from abc import ABC

import os
import torch
import pytorch_lightning as pl
from .models import DotProductPredictionHead
from .models.missl import MISSL
from .models import info_nce_item_beh, info_nce_inter_intent, info_nce_intra_intent
from .utils import recalls_and_ndcgs_for_ks


class RecModel(pl.LightningModule):
    def __init__(self, backbone: pl.LightningModule):
        super().__init__()
        self.backbone = backbone
        self.n_b = backbone.n_b
        self.head_i = DotProductPredictionHead(backbone.d_model, backbone.num_items, self.backbone.embedding.token)
        self.head_b = DotProductPredictionHead(backbone.d_model, backbone.n_b, self.backbone.embedding.typeToken)
        self.cl_ib = info_nce_item_beh
        self.inter_cl_loss = info_nce_inter_intent
        self.intra_cl_loss = info_nce_intra_intent
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, b_seq):
        return self.backbone.forward(input_ids, b_seq)

    def training_step(self, batch, batch_idx):
        if isinstance(self.backbone,  MISSL):
            input_ids, b_seq, labels_i, labels_b = batch['input_ids'], batch['behaviors'], batch['labels_i'], batch['labels_b']
            item_emb_output, type_emb_output, item_type_pos, bs_intent, ba_intent, idx_bs, idx_ba = self.forward(input_ids, b_seq)
            item_outputs = item_emb_output.view(-1, item_emb_output.size(-1))  # BT x H
            type_outputs = type_emb_output.view(-1, type_emb_output.size(-1))  # BT x H
            sample_idxs = []
            for _ in range(input_ids.size(0)):
                label = labels_i[_] > 0
                idx = label.nonzero().squeeze()
                if idx.size() == torch.Size([]):
                    idx = idx.unsqueeze(0)
                if idx.size(0) == 0:
                    continue
                sample_idx = idx[torch.randint(0, idx.size(0), (1,))]
                i = torch.zeros(label.size(), device=label.device)
                i[sample_idx] = 1
                sample_idxs.append(i)

            sample_idxs = (torch.stack(sample_idxs, dim=0) > 0).view(-1).nonzero().squeeze()
            labels_i, labels_b = labels_i.view(-1), labels_b.view(-1)  # BT

            valid = labels_i > 0
            valid_index = valid.nonzero().squeeze()  # M
            valid_outputs_i, valid_outputs_b = item_outputs[valid_index], type_outputs[valid_index]
            valid_labels_i, valid_labels_b = labels_i[valid_index], labels_b[valid_index]
            cl_outputs_i, cl_outputs_b = item_outputs[sample_idxs], type_outputs[sample_idxs]
            valid_logits_i = self.head_i.forward(valid_outputs_i, b_seq)  # M
            valid_logits_b = self.head_b.forward(valid_outputs_b, b_seq)  # M
            try:
                loss_i = self.loss(valid_logits_i, valid_labels_i)
                loss_b = self.loss(valid_logits_b, valid_labels_b)
            except Exception:
                loss_i = 0
                loss_b = 0
            if sample_idxs.size() == torch.Size([]):
                loss_cl_ib = 0
            else:
                loss_cl_ib = self.cl_ib(cl_outputs_i, cl_outputs_b, self.backbone.temp, cl_outputs_i.size(0))
            loss_rec = self.backbone.ni_lambda*loss_i + self.backbone.nb_lambda*loss_b + self.backbone.ib_lambda*loss_cl_ib
            loss_cl_inter = self.backbone.inter_intent_lambda*self.inter_cl_loss(item_type_pos, bs_intent, ba_intent, idx_bs, idx_ba, input_ids != 0, b_seq,
                                                                                 self.backbone.temp, self.backbone.sim, self.backbone.n_b, self.backbone.n_intent,
                                                                                 self.backbone.n_bs_intents, self.backbone.n_ba_intents)
            loss_cl_intra = self.backbone.intra_intent_lambda*self.intra_cl_loss(bs_intent, ba_intent, idx_bs, idx_ba, input_ids != 0, self.backbone.temp, self.backbone.sim,
                                                                                 self.backbone.n_intents, self.backbone.n_bs_intents, self.backbone.n_ba_intents)
            loss = loss_rec + loss_cl_inter + loss_cl_intra
            return {'loss': loss.unsqueeze(0)}

    def training_epoch_end(self, training_step_outputs):
        loss = torch.cat([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        if isinstance(self.backbone, MISSL):
           outputs, _, _, _, _, _, _ = self.backbone(input_ids, b_seq)
           last_outputs = outputs[:, -1, :]
           candidates = batch['candidates'].squeeze()  # B x C
           logits = self.head_i(last_outputs, b_seq, candidates)
           labels = batch['labels'].squeeze()
           metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])
           return metrics

    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0].keys()
        for k in keys:
            tmp = []
            for o in validation_step_outputs:
                tmp.append(o[k])
            self.log(f'Val:{k}', torch.Tensor(tmp).mean())

def multi_hot_embed(masked_index, max_length):
    masked_index = masked_index.view(-1)
    multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device, dtype=torch.long)
    multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
    return multi_hot