import os
import traceback
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

from config import WeakConfig, DebiasConfig
import torch.optim as optim
from Classifier import WeakClassifier, DebiasClassifier, FH
from base_loader import MyDataset, DebiasDataset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from utils import *
import torch
import torch.nn as nn
# import seaborn as sns
# import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification

logger = init_logger(log_name='echo')
seed_everything(1996)

class WeakTrainer(nn.Module):
    '''
    Trainer for Bert-based Model
    '''
    def __init__(self, cfg):
        super(WeakTrainer, self).__init__()
        self.cfg = cfg
        self.model = WeakClassifier(cfg)
        if len(cfg.cudas) > 1:
            self.model = to_multiCUDA(cfg, self.model)
        self.model.cuda()

    def train_weak(self, cfg):
        train_loader = WeaDataset(cfg, self.model.module.tokenizer, True, False)
        test_loader = MyDataset(cfg, self.model.module.tokenizer, False, False)
        optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        # train the model
        best_metrics = 0.0
        MSE = torch.nn.MSELoss()
        CE = torch.nn.CrossEntropyLoss()
        for epoch in range(cfg.epoch):
            total_loss = 0.0
            self.model.train()
            for step, ipt in enumerate(train_loader):
                # ipt = {k: v.to(cfg.device) for k, v in ipt.items()}
                input_ids = ipt['orgin_text'].cuda()
                attention_mask = ipt['attention_mask'].cuda()
                out = self.model(input_ids, attention_mask).logits
                loss = CE(out, ipt['label'].cuda())
                if len(cfg.cudas) > 1:  # check-multi GPUs
                    loss = torch.sum(loss, dim=0)
                # else:
                #     loss = MSE(out.squeeze(), ipt['labels'])
                total_loss += loss.data
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if step % 50 == 0:
                    logger.info("epoch-{}, step-{}/{}, loss:{}".format(epoch, step, train_loader.steps, loss.data))
            # 模型测试
            targets, preds = self.eval_weak(cfg, self.model, test_loader)
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            metric = accuracy_score(targets, preds)
            # else:
            #     metric = test_pearson(targets, preds)
            if metric > best_metrics:
                best_metrics = metric
                self.model.module.save()
            logger.info('====BEST results:{}, resuls:{}, loss:{}====='.format(best_metrics, metric, total_loss))

    def eval_weak(self, cfg, model, loader):
        model.eval()
        targets, preds = [], []
        for step, ipt in enumerate(loader):
            input_ids = ipt['orgin_text'].cuda()
            attention_mask = ipt['attention_mask'].cuda()
            out = model(input_ids, attention_mask).logits
            target = ipt["label"].cpu().detach().numpy()
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
            # else:
            #     pred = out.squeeze().cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
        model.train()
        return targets, preds

    # CD: cross-domain
    def test_CD_weak(self, cfg):
        model = WeakClassifier(cfg).cuda()
        loader = MyDataset(cfg, model.tokenizer, train=False, target=True)
        model.eval()
        targets, preds = self.eval_weak(cfg, model, loader)
        acc = accuracy_score(targets, preds)
        logger.info('============acc on {}-{} is {}================='.format(cfg.dataset, cfg.target, acc))

    def get_topk_attention_sub_keywrods(self):
        assert self.cfg.use_checkpoints == True
        # don't shuffle
        train_loader = MyDataset(self.cfg, self.model.module.tokenizer, train=True, target=False, shuffle=False)
        test_loader = MyDataset(self.cfg, self.model.module.tokenizer, train=False, target=False, shuffle=False)
        attn_model = WeakClassifier(self.cfg).model.cuda()
        train_results = self._get_topk_attention_sub_keywrods_for_each_sample(train_loader, attn_model)
        test_results = self._get_topk_attention_sub_keywrods_for_each_sample(test_loader, attn_model)
        with open('top_att_for_each_sample/train-{}.txt'.format(self.cfg.dataset), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(train_results))
        with open('top_att_for_each_sample/test-{}.txt'.format(self.cfg.dataset), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(test_results))

    def _get_topk_attention_sub_keywrods_for_each_sample(self, loader, attn_model, topk=8):
        SPECIAL_TOKENS = loader.tokenizer.all_special_ids
        PAD_TOKEN = loader.tokenizer.convert_tokens_to_ids(loader.tokenizer.pad_token)
        vocab_size = len(loader.tokenizer)
        results = []
        for step, ipt in tqdm(enumerate(loader)):
            current_words = []
            tokens = ipt['input_ids'].cuda()
            masks = ipt['attention_mask'].cuda()
            with torch.no_grad():
                attention = attn_model(tokens, masks, output_attentions=True).attentions[-1]
            attention = attention.sum(dim=1).cpu()  # sum over attention heads (batch_size, max_len, max_len)
            for i in range(attention.size(0)):  # batch_size
                token_and_score = []
                for j in range(attention.size(-1)):  # max_len
                    token = tokens[i][j].item()
                    if token == PAD_TOKEN:  # token == pad_token
                        break
                    if token in SPECIAL_TOKENS:  # skip special token
                        continue
                    score = attention[i][0][j].item()  # 1st token = CLS token
                    if token not in current_words:
                        token_and_score.append([token, score])
                        current_words.append(token)
                top_tokens = sorted(token_and_score, key=lambda a: a[1], reverse=True)
                top_tokens = [loader.tokenizer.convert_ids_to_tokens(each[0]) for each in top_tokens]
                top_tokens = [each for each in top_tokens if '#' not in each][:topk]
                results.append(' '.join(top_tokens))
        return results

    def get_topk_ig_sub_keywords(self):
        '''
        :return:
        '''
        assert self.cfg.use_checkpoints == True
        train_loader = MyDataset(self.cfg, self.model.module.tokenizer, train=True, target=False, shuffle=False)
        test_loader = MyDataset(self.cfg, self.model.module.tokenizer, train=False, target=False, shuffle=False)
        self.attn_model = WeakClassifier(self.cfg).model.cuda()
        self.attn_model = to_multiCUDA(self.cfg, self.attn_model)
        self.attn_model.eval()
        self.attn_model.zero_grad()
        forward_class = FH(self.attn_model)
        ig = LayerIntegratedGradients(forward_class, self.attn_model.module.bert.embeddings)
        # target class index需要自己定义：除了cls以及sep位置之外，是一个全为pad token的输入，也就是所说的0
        train_results = self._get_topk_ig_sub_keywords_for_each_sample(train_loader, ig)
        test_results = self._get_topk_ig_sub_keywords_for_each_sample(test_loader, ig)
        with open('top_ig_for_each_sample/train-{}.txt'.format(self.cfg.dataset), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(train_results))
        with open('top_ig_for_each_sample/test-{}.txt'.format(self.cfg.dataset), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(test_results))

    def _get_topk_ig_sub_keywords_for_each_sample(self, loader, ig, topk=8):
        results = []
        SPECIAL_TOKENS = loader.tokenizer.all_special_ids
        PAD_TOKEN = loader.tokenizer.convert_tokens_to_ids(loader.tokenizer.pad_token)
        for step, ipt in tqdm(enumerate(loader)):
            input_ids = ipt['orgin_text'].cuda()
            base_ids = ipt['base_text'].cuda()
            label = ipt['label'].cuda()
            attributions, approximation_error = ig.attribute(inputs=input_ids, baselines=base_ids,
                                             target=label, return_convergence_delta=True)
            attributions = torch.abs(attributions).sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            for i in range(attributions.size(0)):  # batch_size
                token_and_score = {}
                for j in range(attributions.size(-1)):  # max_len
                    token = input_ids[i][j].item()
                    if token == PAD_TOKEN:  # token == pad_token
                        break
                    if token in SPECIAL_TOKENS:  # skip special token
                        continue
                    try:
                        score = attributions[i][j].item()
                    except:
                        score = 0.00001
                    # if token not in current_words:
                    if token not in token_and_score.keys():
                        token_and_score[token] = [score]
                    else:
                        token_and_score[token].append(score)
                # you should average a token if it occurs multi-times in the sentence
                top_tokens = sorted(token_and_score.items(), key=lambda a: sum(a[1])/len(a[1]), reverse=True)
                top_tokens = [loader.tokenizer.convert_ids_to_tokens(each[0]) for each in top_tokens]
                top_tokens = [each for each in top_tokens if '#' not in each][:topk]
                results.append(' '.join(top_tokens))
        return results

    def get_global_ig_sub_keywords(self):
        ''' 获取全局的子词的IG得分并进行排序
        :return:
        '''
        assert self.cfg.use_checkpoints == True
        train_loader = MyDataset(self.cfg, self.model.module.tokenizer, train=True, target=False, shuffle=False)
        test_loader = MyDataset(self.cfg, self.model.module.tokenizer, train=False, target=False, shuffle=False)
        self.attn_model = WeakClassifier(self.cfg).model.cuda()
        self.attn_model = to_multiCUDA(self.cfg, self.attn_model)
        self.attn_model.eval()
        self.attn_model.zero_grad()
        forward_class = FH(self.attn_model)
        ig = LayerIntegratedGradients(forward_class, self.attn_model.module.bert.embeddings)
        # target class index需要自己定义：除了cls以及sep位置之外，是一个全为pad token的输入，也就是所说的0
        train_results = self._get_ig_sub_keywords_for_each_sample(train_loader, ig)
        test_results = self._get_ig_sub_keywords_for_each_sample(test_loader, ig)
        with open('global_ig/train-{}.txt'.format(self.cfg.dataset), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(train_results))
        with open('global_ig/test-{}.txt'.format(self.cfg.dataset), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(test_results))

    def _get_ig_sub_keywords_for_each_sample(self, loader, ig):
        results = []
        token_and_score = {}
        SPECIAL_TOKENS = loader.tokenizer.all_special_ids
        PAD_TOKEN = loader.tokenizer.convert_tokens_to_ids(loader.tokenizer.pad_token)
        for step, ipt in tqdm(enumerate(loader)):
            # if step == 2:
            #     break
            input_ids = ipt['orgin_text'].cuda()
            base_ids = ipt['base_text'].cuda()
            label = ipt['label'].cuda()
            attributions, approximation_error = ig.attribute(inputs=input_ids, baselines=base_ids,
                                                             target=label, return_convergence_delta=True)
            attributions = torch.abs(attributions).sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            for i in range(attributions.size(0)):  # batch_size
                for j in range(attributions.size(-1)):  # max_len
                    try:
                        token = input_ids[i][j].item()
                    except:
                        continue
                    if token == PAD_TOKEN:  # token == pad_token
                        break
                    if token in SPECIAL_TOKENS:  # skip special token
                        continue
                    try:
                        score = attributions[i][j].item()
                    except:
                        score = 0.00001
                    # if token not in current_words:
                    if token not in token_and_score.keys():
                        token_and_score[token] = [score]
                    else:
                        token_and_score[token].append(score)
        # you should average a token if it occurs multi-times in the sentence
        top_tokens = sorted(token_and_score.items(), key=lambda a: sum(a[1]) / len(a[1]), reverse=True)
        # top_tokens = [loader.tokenizer.convert_ids_to_tokens(each[0]) for each in top_tokens]
        # top_tokens = [each for each in top_tokens if '#' not in each]
        for token in top_tokens:
            word = loader.tokenizer.convert_ids_to_tokens(token[0])
            score = str(sum(token[1]) / len(token[1]))
            if '#' in word:
                continue
            results.append(word+'\t'+score)
        return results

    def get_global_ig_for_each_sample(self):
        ''' 根据get_global_ig_sub_keywords的结果，为每一个样本生成潜在的候选词表
        :return:
        '''
        with open('global_ig/train-{}.txt'.format(self.cfg.dataset), 'r', encoding='utf-8') as rf:
            train_causal_cands = [each.strip().split('\t')[0] for each in rf.readlines()]
            train_causal_cands = train_causal_cands[:int(len(train_causal_cands)*0.2)]
        with open('global_ig/test-{}.txt'.format(self.cfg.dataset), 'r', encoding='utf-8') as rf:
            test_causal_cands = [each.strip().split('\t')[0] for each in rf.readlines()][:4000]
        train_loader = MyDataset(self.cfg, self.model.module.tokenizer, train=True, target=False, shuffle=False)
        test_loader = MyDataset(self.cfg, self.model.module.tokenizer, train=False, target=False, shuffle=False)
        train_results = []
        for data in train_loader.datas:
            cands = []
            tokens = self.model.module.tokenizer.tokenize(data[0])
            tokens = [each for each in tokens if '#' not in each]
            for token in tokens:
                if len(token) == 1:   # 跳过符号
                    continue
                if token in train_causal_cands and token not in cands:
                    cands.append(token)
            train_results.append(' '.join(cands))
        test_results = []
        for data in test_loader.datas:
            cands = []
            tokens = self.model.module.tokenizer.tokenize(data[0])
            tokens = [each for each in tokens if '#' not in each]
            for token in tokens:
                if token in train_causal_cands:
                    cands.append(token)
            test_results.append(' '.join(cands))
        with open('global_ig_word_group_for_each_sample/train-{}.txt'.format(self.cfg.dataset), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(train_results))
        with open('global_ig_word_group_for_each_sample/test-{}.txt'.format(self.cfg.dataset), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(test_results))

class DebiasTrainer(nn.Module):
    '''
    Trainer for our proposed Debias Model
    '''
    def __init__(self, cfg):
        super(DebiasTrainer, self).__init__()
        self.cfg = cfg

    def train_Debias(self, cfg):
        model = DebiasClassifier(cfg)
        if len(cfg.cudas) > 1:
            model = to_multiCUDA(cfg, model)
        model.cuda()
        train_loader = DebiasDataset(cfg, model.module.tokenizer, True, shuffle=True)
        test_loader = DebiasDataset(cfg, model.module.tokenizer, train=False, target=False, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-8)
        # train the model
        best_metrics = 0.0
        CE_cls = torch.nn.CrossEntropyLoss()
        CE_mlm = torch.nn.CrossEntropyLoss()
        for epoch in range(cfg.epoch):
            total_loss = 0.0
            model.train()
            for step, ipt in enumerate(train_loader):
                try:
                    orgin_logit, mlm_out, clloss = model(ipt)
                    # define loss
                    # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface', 'mr']:
                    loss_cls = CE_cls(orgin_logit, ipt['ce_label'].cuda()).cpu()
                    # loss_mlm = CE_mlm(mlm_out, ipt['masked_label'].cuda()).cpu()
                    if len(cfg.cudas) > 1:  # check-multi GPUs
                        loss_cls = torch.sum(loss_cls, dim=0)
                        clloss = torch.sum(clloss, dim=0)
                        # loss_mlm = torch.sum(loss_mlm, dim=0)
                    # total_loss = loss_cls + cfg.lambda1*loss_mlm + cfg.lambda2*clloss
                    total_loss = loss_cls + cfg.lambda2*clloss
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    if step % 50 == 0:
                        logger.info("epoch-{}, step-{}/{}, loss:{}, loss CE:{} loss mlm:{}, loss cl:{}".format(epoch, step,
                                        train_loader.steps, total_loss.data, loss_cls.data, 0, clloss))
                except:
                    traceback.print_exc()
                    continue
            # 模型测试
            targets, preds = self.eval_Debias(model, test_loader)
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            metric = accuracy_score(targets, preds)
            # else:
            #     metric = test_pearson(targets, preds)
            if metric > best_metrics:
                best_metrics = metric
                model.module.save()
            logger.info('====BEST results:{}, resuls:{}, loss:{}====='.format(best_metrics, metric, total_loss))

    def eval_Debias(self, model, loader):
        model.eval()
        targets, preds = [], []
        for step, ipt in enumerate(loader):
            orgin_logit, _, _ = model(ipt)
            target = ipt["ce_label"].cpu().detach().numpy()
            pred = torch.max(orgin_logit, dim=-1)[1].cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
        model.train()
        return targets, preds

    # CD: cross-domain
    def test_CD_debias(self, cfg):
        model_dir_name = '{}_on_{}_shortcutMLM_CL_{}_{}'.format(self.cfg.base_model, self.cfg.dataset,
                                                                self.cfg.lambda1, self.cfg.lambda2)
        save_path = 'save_models/cl_classifier/' + model_dir_name
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
        model = AutoModelForSequenceClassification.from_pretrained(save_path).cuda()
        loader = MyDataset(cfg, tokenizer, train=False, target=True)
        model.eval()
        targets, preds = [], []
        for step, ipt in enumerate(loader):
            input_ids = ipt['orgin_text'].cuda()
            attention_mask = ipt['attention_mask'].cuda()
            out = model(input_ids, attention_mask).logits
            target = ipt["label"].cpu().detach().numpy()
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
            # else:
            #     pred = out.squeeze().cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
        acc = accuracy_score(targets, preds)
        logger.info('============acc on {}-{} is {}================='.format(cfg.dataset, cfg.target, acc))
        # model = DebiasClassifier(cfg, is_CD=True).cuda()
        # model.eval()
        # target_loader = DebiasDataset(cfg, model.tokenizer, train=False, target=True)
        # targets, preds = [], []
        # for step, ipt in enumerate(target_loader):
        #     orgin_logit, _, _ = model(ipt)
        #     target = ipt["ce_label"].cpu().detach().numpy()
        #     pred = torch.max(orgin_logit, dim=-1)[1].cpu().detach().numpy()
        #     targets.extend(list(target))
        #     preds.extend(list(pred))
        # acc = accuracy_score(targets, preds)
        # logger.info('============acc on {}-{} is {}================='.format(cfg.dataset, cfg.target, acc))

    def LFR(self, cfg):
        '''  test the Label Flipping Rate for the word-groups augmentation
        :return:
        '''
        # 1. load model
        # model_dir_name = '{}_on_{}_shortcutMLM_CL_{}_{}'.format(self.cfg.base_model, self.cfg.dataset,
        #                                                         self.cfg.lambda1, self.cfg.lambda2)
        # save_path = 'save_models/cl_classifier/' + model_dir_name
        model_dir_name = '{}_on_{}'.format(cfg.base_model, cfg.dataset)
        save_path = 'save_models/weakclassifier/' + model_dir_name
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
        model = AutoModelForSequenceClassification.from_pretrained(save_path).cuda()
        # 2. prepare dataset, just train loader used
        train_loader = DebiasDataset(cfg, tokenizer, True, shuffle=False)
        # eval the model
        targets, preds, cad_preds0, cad_preds1, cad_preds2 = [], [], [], [], []
        for step, ipt in enumerate(train_loader):
            input_ids = ipt['orgin_text'].cuda()
            attention_mask = ipt['attention_mask'].cuda()
            out = model(input_ids, attention_mask).logits
            cad_out0 = model(ipt['cad_text_0'].cuda(), ipt['cad_attention_mask_0'].cuda()).logits
            cad_out1 = model(ipt['cad_text_1'].cuda(), ipt['cad_attention_mask_1'].cuda()).logits
            cad_out2 = model(ipt['cad_text_2'].cuda(), ipt['cad_attention_mask_2'].cuda()).logits
            target = ipt["ce_label"].cpu().detach().numpy()
            # if cfg.dataset in ['foods', 'sst2', 'imdb', 'telephone', 'letters', 'facetoface']:
            pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
            cad_pred0 = torch.max(cad_out0, dim=-1)[1].cpu().detach().numpy()
            cad_pred1 = torch.max(cad_out1, dim=-1)[1].cpu().detach().numpy()
            cad_pred2 = torch.max(cad_out2, dim=-1)[1].cpu().detach().numpy()
            # else:
            #     pred = out.squeeze().cpu().detach().numpy()
            targets.extend(list(target))
            preds.extend(list(pred))
            cad_preds0.extend(list(cad_pred0))
            cad_preds1.extend(list(cad_pred1))
            cad_preds2.extend(list(cad_pred2))
        acc = accuracy_score(targets, cad_preds0)
        logger.info('LFR for a single word-group L=1 is {}'.format(1-acc))
        # 3. 计算word-group的反转结果，只要有一个翻转即可
        flip_counts = 0
        for t, p1, p2, p3 in zip(targets, cad_preds0, cad_preds1, cad_preds2):
            if t == p1 and t == p2 and t == p3:
                continue
            else:
                flip_counts += 1
        logger.info('LFR for word-groups L=3 is {}'.format(flip_counts/len(targets)))


if __name__ == '__main__':
    weak_cfg = WeakConfig()
    for data in ['foods', 'sst2', 'mr', 'kindle']:
        weak_cfg.use_checkpoints = False
        weak_cfg.dataset = data
        weak_cfg.batch = 128
        trainer = WeakTrainer(weak_cfg)
        trainer.train_weak(weak_cfg)
        for target in ['foods', 'sst2', 'mr', 'kindle']:
            weak_cfg.target = target
            weak_cfg.use_checkpoints = True
            trainer.test_CD_weak(weak_cfg)

    trainer.get_global_ig_sub_keywords()
    trainer.get_global_ig_for_each_sample()
    # exit(111)
    # for data in ['foods', 'sst2', 'mr', 'kindle']:
    #     weak_cfg.dataset = data
    #     weak_cfg.use_checkpoints = False
    #     trainer = WeakTrainer(weak_cfg)
    #     # trainer.train_weak(weak_cfg)
    #     for dataset in ['foods', 'sst2', 'mr', 'kindle']:
    #         weak_cfg.target = dataset
    #         weak_cfg.use_checkpoints = True
    #         trainer.test_CD_weak(weak_cfg)
    # exit(111)
    # for dataset in ['mr']:
    #     weak_cfg.dataset = dataset
    #     trainer.get_topk_ig_sub_keywords()
    # for dataset in ['foods', 'sst2', 'imdb_s', 'mr', 'kindle']:
    #     weak_cfg.target = dataset
    #     logger.info('test {} on target {}'.format(weak_cfg.dataset, dataset))
    #     trainer.test_CD_weak(weak_cfg)
    #     # trainer.get_topk_attention_sub_keywrods()
    cfg = DebiasConfig()
    for lll in [0.01]:
        cfg.lambda1 = lll
        cfg.lambda2 = lll
        for data in ['books', 'dvd', 'electronics', 'kitchen']:
            cfg.dataset = data
            cfg.batch = 32
            trainer = DebiasTrainer(cfg)
            trainer.train_Debias(cfg)
            trainer.LFR(cfg)
            for dataset in ['Davids', 'OffEval', 'Abusive', 'ToxicTweets']:
                cfg.target = dataset
                cfg.batch = 64
                logger.info('test {} on target {}'.format(cfg.dataset, dataset))
                trainer.test_CD_debias(cfg)
