import torch.nn as nn
from transformers import *
from transformers import BertForSequenceClassification
import torch
from os.path import join, exists
import torch.functional as F
import os
import json

def load_backbone(cfg):
    model_dir_name = '{}_on_{}'.format(cfg.base_model, cfg.dataset)
    save_path = 'save_models/weakclassifier/' + model_dir_name
    if cfg.use_checkpoints == False:
        print('load from official checkpoints')
        backbone = AutoModelForSequenceClassification.from_pretrained(cfg.bert_path, num_labels=cfg.nclass)
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
    else:
        print('load from {}'.format(save_path))
        backbone = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=cfg.nclass)
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)

    return backbone, tokenizer

class WeakClassifier(nn.Module):
    def __init__(self, cfg):
        super(WeakClassifier, self).__init__()
        self.cfg = cfg
        self.model, self.tokenizer = load_backbone(self.cfg)

    def forward(self, input_ids, attention_mask=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True,
                         output_hidden_states=True, return_dict=True)
        return out

    def get_grad(self, input_ids, segment_ids=None, input_mask=None, label_ids=None,
                 tar_layer=None, one_batch_att=None, pred_label=None):
        '''
        :param input_ids:  input ids
        :param segment_ids:  token_type_ids
        :param input_mask:  attention masks
        :param label_ids:
        :param tar_layer: the layer of the attention head
        :param one_batch_att: the attention score of the target layer
        :param pred_label: the label predicted by baseline language model rather than the ground truth
        :return:
        '''
        _, pooled_output, att_score = self.model(
            input_ids, segment_ids, input_mask, output_all_encoded_layers=False,
            tar_layer=tar_layer, tmp_score=one_batch_att)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        prob = torch.softmax(logits, dim=-1)
        tar_prob = prob[:, label_ids[0]]
        if one_batch_att is None:
            return att_score[0], logits
        else:
            #gradient = torch.autograd.grad(torch.unbind(prob[:, labels[0]]), tmp_score)
            gradient = torch.autograd.grad(torch.unbind(prob[:, pred_label]), one_batch_att)
            return tar_prob, gradient[0]

    def get_causal_js(self, xs, cands, x_masks, cand_masks):
        # 利用已经训练好的弱分类器进行 原样本-反事实样本 之间的散度计算
        '''
        :param x: [原样本repaet的扩充ids]
        :param x_s: [候选因果特征词, 因果替换反事实样本的ids]
        :return: [候选因果特征词, 替换特征词带来的模型预测的js散度的变化]
        '''
        xs = xs.to(self.cfg.device)
        out = self.model(input_ids=xs, attention_mask=x_masks, output_hidden_states=True, return_dict=False)[0]
        out_ = self.model(input_ids=cands, attention_mask=cand_masks, output_hidden_states=True, return_dict=False)[0]
        jss = []
        for i in range(out_.shape[0]):
            js = self._js_div(out[i], out_[i])
            jss.append(float(js.detach().cpu())*100)
        return jss

    def _js_div(self, p_output, q_output, get_softmax=True):
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = torch.softmax(p_output, dim=-1)
            q_output = torch.softmax(q_output, dim=-1)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

    def save(self):
        model_dir_name = '{}_on_{}'.format(self.cfg.base_model, self.cfg.dataset)
        save_path = 'save_models/weakclassifier/' + model_dir_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_pretrained(save_path)

class DebiasClassifier(nn.Module):
    def __init__(self, cfg, is_CD=False):
        super(DebiasClassifier, self).__init__()
        self.cfg = cfg
        model_dir_name = '{}_on_{}_shortcutMLM_CL_{}_{}'.format(self.cfg.base_model, self.cfg.dataset,
                                                                self.cfg.lambda1, self.cfg.lambda2)
        self.save_path = 'save_models/cl_classifier/' + model_dir_name
        if is_CD:
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                self.save_path, num_labels=self.cfg.nclass)
        else:
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                    self.cfg.bert_path, num_labels=self.cfg.nclass)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
        self.net_ssl = nn.Sequential(  # self-supervision layer for MLM token prediction
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, len(self.tokenizer.get_vocab())),
        )
        self.CL_dense = nn.Linear(768, 128)   # self-supervision layer for CL loss as SimCLR
        self.W = nn.Linear(128*3, 3)   # 3 是cad样本的组数，用于获取attention value
        self.dropout = nn.Dropout(0.1)

    def forward(self, ipt, idx=3):
        '''
        :param ipt:
        :param idx: 使用几个cad样本做增强
        :return:
        '''
        orgin_text = ipt['orgin_text'].cuda()
        masked_text = ipt['masked_text_0'].cuda()
        attention_mask = ipt['attention_mask'].cuda()
        cad_clss, cad_indeces = [], []
        for i in range(idx):
            cad_text = ipt['cad_text_{}'.format(i)].cuda()
            # cad_index = ipt['cad_index_{}'.format(i)].cuda()
            cad_attention_mask = ipt['cad_attention_mask_{}'.format(i)].cuda()
            cad_indeces.append(ipt['cad_index_{}'.format(i)].cuda())
            cad_out = self.backbone(cad_text, cad_attention_mask, output_hidden_states=True)
            cad_cls = cad_out.hidden_states[-1][:, 0, :]
            cad_cls = self.dropout(cad_cls)
            cad_cls = self.CL_dense(cad_cls)
            cad_clss.append(cad_cls)
        orgin_out = self.backbone(orgin_text, attention_mask, output_hidden_states=True)
        masked_out = self.backbone(masked_text, attention_mask, output_hidden_states=True)
        orgin_logit, orgin_cls = orgin_out.logits, orgin_out.hidden_states[-1][:,0,:]
        masked_cls, masked_out = masked_out.hidden_states[-1][:,0,:], masked_out.hidden_states[-1]
        # del cad_out
        # dropout
        orgin_cls = self.dropout(orgin_cls)
        masked_cls = self.dropout(masked_cls)
        # for mlm out
        # mlm_out = self.net_ssl(masked_out).permute(0, 2, 1)
        # for CL linear as SimCLR,
        orgin_cls = self.CL_dense(orgin_cls)
        masked_cls = self.CL_dense(masked_cls)
        clloss = self.CLLoss(orgin_cls, masked_cls, cad_clss, cad_indeces)
        return orgin_logit, None, clloss

    def CLLoss2(self, z1, z2, z3, cad_index):
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        z3 = torch.nn.functional.normalize(z3, dim=1)
        delta = 1
        item1 = torch.nn.functional.cosine_similarity(z1, z2, -1, 1e-8)
        item1 = torch.mean(item1)
        item2 = torch.nn.functional.cosine_similarity(z1, z3, -1, 1e-8)  # 降低其相似度
        summ = torch.count_nonzero(cad_index, dim=-1)
        if summ == 0:
            item2 = 0
        else:
            z3_index = torch.where(cad_index == 0, 0, 1)
            item2 = item2 * z3_index  # batch
            item2 = torch.mean(torch.sum(item2, dim=-1))
        loss = delta - item1 + item2
        # loss = delta+item2
        loss = max(loss - loss, loss)  # 注意相似度
        # loss = max(0, delta+item2)
        return loss

    def CLLoss(self, z1, z2, cads, cad_index):
        '''
        :param z1:  原样本的表示
        :param z2:  masked shortcut的样本表示
        :param z3:  CAD的样本表示
        :param z3_index:  CAD的index，因为并不是所有的样本都有CAD
        :return: 最终的联合损失，参照 Causally Contrastive Learning for Robust Text Classification
        '''
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        delta = 1
        item1 = torch.nn.functional.cosine_similarity(z1, z2, -1, 1e-8)
        item1 = torch.mean(item1)
        cad_losses, cad_vecs = [], []
        no_zero_summ = 0
        for i in range(len(cads)):
            z3 = torch.nn.functional.normalize(cads[i], dim=1)
            z3_index = cad_index[i]
            item2 = torch.nn.functional.cosine_similarity(z1, z3, -1, 1e-8)  # 降低其相似度
            if torch.sum(z3_index) != 0:
                summ = torch.count_nonzero(z3_index, dim=-1)
                no_zero_summ += summ
                z3_index = torch.where(z3_index == 0, 0, 1)
                item2 = item2 * z3_index  # batch
            cad_losses.append(item2.unsqueeze(-1))
        cad_losses = torch.cat(cad_losses, dim=-1)  # batch * 3
        cad_vecs = torch.cat(cads, dim=-1)
        cad_att = self.W(cad_vecs)  # batch*3  3 for cad_samples
        cad_att = torch.softmax(cad_att, dim=-1)  # 注意力归一化
        cad_loss = cad_att * cad_losses
        item2 = torch.mean(torch.sum(cad_loss, dim=-1))
        loss = delta-item1+item2
        # loss = delta+item2
        loss = max(loss-loss, loss)  # 注意相似度
        # loss = max(0, delta+item2)
        return loss

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.backbone.save_pretrained(self.save_path)

# Just for IG parallel used
class FH(nn.Module):
    def __init__(self, model):
        super(FH, self).__init__()
        self.model = model
        self.device_ids = model.device_ids

    def forward(self, input_ids):
        logits = self.model(input_ids).logits
        return logits