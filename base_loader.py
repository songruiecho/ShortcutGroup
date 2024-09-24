import copy
import math
import os
import json
import random
import traceback

import pandas as pd
from transformers import *
from utils import *

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
import numpy as np
from Classifier import *
from tqdm import tqdm
import pickle
from config import DebiasConfig, WeakConfig

class BaseDataset():
    def __init__(self, cfg, tokenizer, train, target=False, shuffle=False):
        super(BaseDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.shuffle = shuffle
        self.target = target
        if target:
            self.dataset = cfg.target
        else:
            self.dataset = cfg.dataset
        self.train = train
        self.ipts = 0
        self.init()
        self.steps = self.get_steps()

    def init(self):
        self.datas = self.load_data()
        self.data_iter = iter(self.datas)
        print('====init DataIter {} with Steps {}==='.format(self.dataset, self.get_steps()))

    def __iter__(self):
        return self

    def get_steps(self):
        return len(self.datas) // self.cfg.batch

    def reset(self):
        self.data_iter = iter(self.datas)

    def load_data(self):
        datas = []
        if self.dataset == 'foods':
            if self.train:
                file = open('dataset/foods_train.txt', 'r', encoding='utf-8')
            else:
                file = open('dataset/foods_test.txt', 'r', encoding='utf-8')
            for line in file.readlines():
                line = line.strip().split(':')
                if int(line[1]) == 1:  # pre-defined class 0
                    label = 0
                elif int(line[1]) == 5:  # pre-defined class 1
                    label = 1
                else:
                    continue
                datas.append([line[0].lower(), label])
        if self.dataset == 'sst2':
            if self.train:
                file = open('dataset/sst2_train.tsv', 'r', encoding='utf-8')
            else:
                file = open('dataset/sst2_dev.tsv', 'r', encoding='utf-8')
            for line in file.readlines():
                line = line.strip().split('\t')
                if self.train:
                    datas.append([line[0].lower(), int(line[1])])
                else:
                    datas.append([line[0].lower(), int(line[1])])
        if self.dataset == 'imdb':
            if self.train:
                frame = pd.read_csv('dataset/imdb_train.csv')
            else:
                frame = pd.read_csv('dataset/imdb_train.csv')
            for t, l in zip(frame['review'].values, frame['label'].values):
                datas.append([t.strip().lower(), int(l)])
        if self.dataset in ['telephone', 'letters', 'facetoface']:
            labeldict = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
            if self.train:
                frame = pd.read_csv('dataset/train-{}.csv'.format(self.dataset))
            else:
                frame = pd.read_csv('dataset/dev-{}.csv'.format(self.dataset))
            s1, s2, ls = list(frame['sentence1'].values), list(frame['sentence2'].values), \
                         list(frame['gold_label'].values)
            for i in range(len(s1)):
                if ls[i] not in labeldict.keys():
                    continue
                try:
                    datas.append([s1[i].lower() + self.tokenizer.sep_token + s2[i].lower(), labeldict[ls[i]]])
                except:
                    continue
        if self.dataset in ['images', 'headlines', 'MSRvid']:
            if self.train:
                frame = pd.read_csv('dataset/train-{}.csv'.format(self.cfg.dataset))
            else:
                frame = pd.read_csv('dataset/dev-{}.csv'.format(self.cfg.dataset))
            s1, s2, ls = frame['sentence1'].values, frame['sentence2'].values, frame['score'].values
            for i in range(len(s1)):
                datas.append([s1[i].lower() + self.tokenizer.sep_token + s2[i].lower(), float(ls[i])])
        if self.dataset in ['mr', 'imdb_s', 'kindle', 'amazon']:
            if self.train:
                file = open('dataset/{}_train.txt'.format(self.dataset), 'r', encoding='utf-8')
            else:
                file = open('dataset/{}_test.txt'.format(self.dataset), 'r', encoding='utf-8')
            lists = [each for each in file.readlines()]
            if self.shuffle:  # 是否混洗
                random.shuffle(lists)
            for line in lists:
                line = line.strip().split()
                datas.append([' '.join(line[1:]), int(line[0])])
        if self.dataset in ['Davids', 'HatEval', 'OffEval', 'Abusive', 'StormW', 'ToxicTweets',
                            'books', 'dvd', 'electronics', 'kitchen']:
            if self.train:
                file = open('dataset/{}_train.txt'.format(self.dataset), 'r', encoding='utf-8')
            else:
                file = open('dataset/{}_test.txt'.format(self.dataset), 'r', encoding='utf-8')
            lists = [each for each in file.readlines()]
            if self.shuffle:  # 是否混洗
                random.shuffle(lists)
            for line in lists:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                datas.append([line[1], int(line[0])])
        return datas

    # the followings need to be rewrote
    def __next__(self):
        pass

    def get_batch(self):
        pass

class MyDataset(BaseDataset):
    def __init__(self, cfg, tokenizer, train, target=False, shuffle=True):
        super(MyDataset, self).__init__(cfg, tokenizer, train=train, target=target, shuffle=shuffle)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.train = train
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.ipts = 0
        self.steps = self.get_steps()

    def get_batch(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.cfg.batch:
                break
        if len(batch_data) < 1:
            return None
        # base_texts is the x' of IntegratedGrads Method, all pad_tokens
        texts, labels, base_texts, masks  = [], [], [], []
        for data in batch_data:
            text, label = data[0], data[1]
            tokens = self.tokenizer.tokenize(text)[:self.cfg.max_len]
            base_tokens = [self.tokenizer.pad_token] * len(tokens)
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            base_tokens = [self.tokenizer.cls_token] + base_tokens + [self.tokenizer.sep_token]
            mask = [1]*len(base_tokens)
            base_tokens = self.tokenizer.convert_tokens_to_ids(base_tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            texts.append(tokens)
            base_texts.append(base_tokens)
            labels.append(label)
            masks.append(mask)
        max_token_len = max(len(each) for each in texts)
        texts = self._padding(max_token_len, texts, self.PAD_ID)
        base_texts = self._padding(max_token_len, base_texts, self.PAD_ID)
        masks = self._padding(max_token_len, masks, 0)
        return {
            'orgin_text': torch.LongTensor(texts),
            'base_text': torch.LongTensor(base_texts),
            'label': torch.LongTensor(labels),
            'attention_mask': torch.LongTensor(masks)
        }

    def _padding(self, max_token_len, texts, pad_token):
        resutls = []
        for text in texts:
            text = text[:self.cfg.max_len]
            if len(text) < max_token_len:
                text = text + [pad_token]*(max_token_len-len(text))
            else:
                text = text[:max_token_len]
            resutls.append(text)
        return resutls

    # next operation for Python iterator, it's necessary for your own iter
    def __next__(self):
        if self.ipts is None:
            self.reset()
        self.ipts = self.get_batch()  # each iter get a batch
        if self.ipts is None:
            raise StopIteration
        else:
            return self.ipts

class DebiasDataset(BaseDataset):
    # dataset for
    def __init__(self, cfg, tokenizer, train, target=False, shuffle=True):
        super(DebiasDataset, self).__init__(cfg, tokenizer, train=train, target=target, shuffle=False)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.train = train
        self.target = target
        self.shuffle = shuffle
        self.ipts = 0
        self.steps = self.get_steps()
        self.get_word_group()
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.MASK_TOKEN = self.tokenizer.mask_token
        self.SEP_TOKEN = self.tokenizer.sep_token

    def get_word_group(self, mode='mode1'):
        '''
        :param mode: 加载模式。mode1表示加载最高的。mode2表示加载前2高的。mode3加载前三。它用于多投票机制
        :return:
        '''
        # 加载潜在捷径词表
        with open('global_ig_word_group_for_each_sample/train-{}.txt'.format(self.cfg.dataset), 'r', encoding='utf-8') as rf:
            self.train_causal_cands = [each.strip().split() for each in rf.readlines()]
        with open('global_ig_word_group_for_each_sample/test-{}.txt'.format(self.cfg.dataset), 'r', encoding='utf-8') as rf:
            self.test_causal_cands = [each.strip().split() for each in rf.readlines()]
        with open('global_ig_word_group_for_each_sample/test-{}.txt'.format(self.cfg.target), 'r', encoding='utf-8') as rf:
            self.target_causal_cands = [each.strip().split() for each in rf.readlines()]
        with open('antony.json', 'rb') as rf:
            self.ANTONY = json.load(rf)
        if self.target:
            with open('global_ig_word_group_for_each_sample/test-{}.json'.format(self.cfg.target), 'r') as f:
                self.WGs = json.load(f)
            self.SCs = self.target_causal_cands
        else:
            if self.train:
                with open('global_ig_word_group_for_each_sample/train-{}.json'.format(self.cfg.dataset), 'r') as f:
                    self.WGs = json.load(f)
                self.SCs = self.train_causal_cands
            else:
                with open('global_ig_word_group_for_each_sample/test-{}.json'.format(self.cfg.dataset), 'r') as f:
                    self.WGs = json.load(f)
                self.SCs = self.test_causal_cands
        self.SCs.append([])
        self.SCs = self.SCs[:len(self.datas)]
        # print(len(self.WGs), len(self.datas), len(self.SCs))
        assert len(self.WGs) == len(self.datas) == len(self.SCs)
        Datas = []
        for data, WG, SC in zip(self.datas, self.WGs, self.SCs):
            best_WG = list(WG)[:3]    # 取前三WG进行词组投票
            causals = []
            for each in best_WG:
                causals.extend(each.split())
            shortcuts = list(set(SC) - set(causals))   # 将WG中的词排除，剩余的候选词作为shortcut
            if len(best_WG) == 0:
                groups = ['']
                scores = [0]
            else:
                groups, scores = [], []
                for each in best_WG:
                    groups.append(each)
                    scores.append(WG[each])
            data.append(groups)
            data.append(scores)
            data.append(shortcuts)
            Datas.append(data)
        self.datas = Datas
        if self.shuffle:
            random.shuffle(self.datas)
            self.data_iter = iter(self.datas)
            self.data_iter = iter(self.datas)
        print('============load word group finished!!!==========='.format())


    def _mask_token(self, tokens, causal_term, shortcuts, shortcut_mask_ratio=0.5, ordinary_mask_ratio=0.15, label=1):
        ''' 基于causal_term执行原样本的反事实mask
        :param tokens:
        :param causal_term:
        :param shortcut_mask_ratio:
        :param ordinary_mask_ratio:
        :param label:
        :return:
        '''
        masked_token = []  # 用于还原shortcut，并作为数据增强的正例
        token_labels = []  # 返回MLM的labels，需要用语言模型进行还原
        cad_tokens = []     # 反事实增强之后的样本,这个样本是根据因果词的翻转得到的
        for token in tokens:
            if token in causal_term.split():
                if token in self.ANTONY.keys():
                    cad_token = self.tokenizer.tokenize(self.ANTONY[token])
                    cad_tokens.extend(cad_token)
                else:
                    cad_token = self.MASK_TOKEN
                    cad_tokens.append(cad_token)
            else:
                cad_tokens.append(token)
            if token in shortcuts:
                mmm = random.random()
                if mmm < shortcut_mask_ratio:
                    masked_token.append(self.MASK_TOKEN)
                    token_labels.append(self.tokenizer.convert_tokens_to_ids(token))
                else:
                    masked_token.append(token)
                    token_labels.append(-100)
            else:
                masked_token.append(token)
                token_labels.append(-100)
        return masked_token, token_labels, cad_tokens

    def _padding(self, max_token_len, texts, pad_token):
        resutls = []
        for text in texts:
            text = text[:self.cfg.max_len]
            if len(text) < max_token_len:
                text = text + [pad_token]*(max_token_len-len(text))
            else:
                text = text[:max_token_len]
            resutls.append(text)
        return resutls

    def get_batch(self):
        return_dict = {}
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.cfg.batch:
                break
        if len(batch_data) < 1:
            return None
        texts, ce_labels, att_masks = [], [], []
        Cad_Texts, Cad_Indices, Cad_Att_Masks, Masked_Texts, Masked_Labels = [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]
        for data in batch_data:
            text, label, causal_terms, scores, shortcuts = data[0], data[1], data[2], data[3], data[4]
            causal_terms = (causal_terms*3)[:3]
            scores = (scores*3)[:3]   # 防止有些词组数量不足3
            tokens = self.tokenizer.tokenize(text)[:self.cfg.max_len]
            idx = 0   # for the i-th cad_sample
            for causal_term, score in zip(causal_terms, scores):
                masked_token, token_labels, cad_tokens = self._mask_token(tokens, causal_term, shortcuts, label=label)
                masked_token = self.tokenizer.convert_tokens_to_ids(masked_token)
                Masked_Texts[idx].append(masked_token)
                Masked_Labels[idx].append([-100]+token_labels + [-100])
                # CAD样本
                if score > 0:  # 说明有反事实增强
                    cad_tokens = [self.tokenizer.cls_token] + cad_tokens + [self.tokenizer.sep_token]
                else:
                    cad_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                # if self.cfg.dataset in ['Davids', 'OffEval', 'StormW', 'Abusive']:
                #     if label == 0 or label == '0':
                #         score = 0    # 对于toxic，只识别正样本
                cad_tokens = self.tokenizer.convert_tokens_to_ids(cad_tokens)
                Cad_Texts[idx].append(cad_tokens)
                mask = [1] * len(cad_tokens)
                Cad_Att_Masks[idx].append(mask)
                Cad_Indices[idx].append(score)  # 用于mask判断是否有CAD
                # idx 增加
                idx += 1
            # print(len(Cad_Texts[0]), len(Cad_Texts[1]), len(Cad_Texts[2]))
            # 原样本
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(tokens)
            att_masks.append(mask)
            texts.append(tokens)
            ce_labels.append(label)
        # mask and padding
        max_token_len = max(len(each) for each in texts)
        texts = self._padding(max_token_len, texts, self.PAD_ID)
        att_masks = self._padding(max_token_len, att_masks, 0)
        return_dict['orgin_text'] = torch.LongTensor(texts)
        return_dict['attention_mask'] = torch.LongTensor(att_masks)
        return_dict['ce_label'] = torch.LongTensor(ce_labels)
        for i in range(len(Cad_Texts)):
            max_cad_len = max(len(each) for each in Cad_Texts[i])
            cad_indices = Cad_Indices[i]
            masked_texts = self._padding(max_token_len, Masked_Texts[i], self.PAD_ID)
            masked_labels = self._padding(max_token_len, Masked_Labels[i], -100)
            cad_texts = self._padding(max_cad_len, Cad_Texts[i], self.PAD_ID)
            cad_att_masks = self._padding(max_cad_len, Cad_Att_Masks[i], 0)
            return_dict['masked_text_{}'.format(i)] = torch.LongTensor(masked_texts)
            return_dict['masked_label_{}'.format(i)] = torch.LongTensor(masked_labels)
            return_dict['cad_text_{}'.format(i)] = torch.LongTensor(cad_texts)
            return_dict['cad_index_{}'.format(i)] = torch.LongTensor(cad_indices)
            return_dict['cad_attention_mask_{}'.format(i)] = torch.LongTensor(cad_att_masks)
        return return_dict

    def __next__(self):
        if self.ipts is None:
            self.reset()
        self.ipts = self.get_batch()  # each iter get a batch
        if self.ipts is None:
            raise StopIteration
        else:
            return self.ipts

class AttackDataset():
    def __init__(self, cfg, tokenizer, attacker='textfooler', max_attack_tokens=3, dataset='mr', shuffle=False):
        '''
        :param cfg:
        :param tokenizer:
        :param attacker: 所使用的文本攻击的基线方法
        :param max_attack_tokens: 最大的攻击token数量。我们在1，2，3之间进行测试
        :param shuffle:
        '''
        super(AttackDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.shuffle = shuffle
        self.attacker = attacker
        self.dataset = dataset
        self.max_attack_tokens = max_attack_tokens
        self.init()
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.ipts = 0
        self.steps = self.get_steps()

    def init(self):
        self.load_data()
        self.data_iter = iter(self.datas)
        print('====init DataIter {} with Steps {}==='.format(self.dataset, self.get_steps()))

    def __iter__(self):
        return self

    def get_steps(self):
        return len(self.datas) // self.cfg.batch

    def reset(self):
        self.data_iter = iter(self.datas)

    def load_data(self):
        frame = pd.read_csv('attack_data/{}_attack_{}.csv'.format(self.attacker, self.dataset))
        original_texts = list(frame['original_text'].values)
        perturbed_texts = list(frame['perturbed_text'].values)
        labels = list(frame['original_output'].values)
        self.datas = []
        assert len(original_texts) == len(perturbed_texts) == len(labels)
        for ori_text, per_text, label in zip(original_texts, perturbed_texts, labels):
            attacked_numbers = 0
            ori_tokens, per_tokens = ori_text.split(), per_text.split()
            text = copy.deepcopy(ori_tokens)
            for i, ori_token in enumerate(ori_tokens):
                if '[[' in ori_token:
                    text[i] = per_tokens[i][2:-2]
                    attacked_numbers += 1
                if attacked_numbers >= self.max_attack_tokens:
                    break
            text = ' '.join(text)
            self.datas.append([text, label])

    def _padding(self, max_token_len, texts, pad_token):
        resutls = []
        for text in texts:
            text = text[:self.cfg.max_len]
            if len(text) < max_token_len:
                text = text + [pad_token]*(max_token_len-len(text))
            else:
                text = text[:max_token_len]
            resutls.append(text)
        return resutls

    def get_batch(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.cfg.batch:
                break
        if len(batch_data) < 1:
            return None
        # base_texts is the x' of IntegratedGrads Method, all pad_tokens
        texts, labels, base_texts, masks  = [], [], [], []
        for data in batch_data:
            text, label = data[0], data[1]
            tokens = self.tokenizer.tokenize(text)[:self.cfg.max_len]
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            texts.append(tokens)
            labels.append(label)
            mask = [1]*len(tokens)
            masks.append(mask)
        max_token_len = max(len(each) for each in texts)
        texts = self._padding(max_token_len, texts, self.PAD_ID)
        masks = self._padding(max_token_len, masks, 0)
        return {
            'orgin_text': torch.LongTensor(texts),
            'label': torch.LongTensor(labels),
            'attention_mask': torch.LongTensor(masks)
        }

    def __next__(self):
        if self.ipts is None:
            self.reset()
        self.ipts = self.get_batch()  # each iter get a batch
        if self.ipts is None:
            raise StopIteration
        else:
            return self.ipts

class GenderDataset():
    def __init__(self, cfg, tokenizer, dataset, shuffle=False):
        '''
        :param cfg:
        :param tokenizer:
        :param attacker: 所使用的文本攻击的基线方法
        :param max_attack_tokens: 最大的攻击token数量。我们在1，2，3之间进行测试
        :param shuffle:
        '''
        super(GenderDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.dataset = dataset
        self.shuffle = shuffle
        self.init()
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.ipts = 0
        self.steps = self.get_steps()

    def init(self):
        self.load_data()
        self.data_iter = iter(self.datas)
        print('====init DataIter {} with Steps {}==='.format(self.dataset, self.get_steps()))

    def __iter__(self):
        return self

    def get_steps(self):
        return len(self.datas) // self.cfg.batch

    def reset(self):
        self.data_iter = iter(self.datas)

    def load_data(self):
        self.datas = []
        with open('gender_data/gender.txt', 'r', encoding='utf-8') as rf:
            genders = [each.strip().split('-') for each in rf.readlines()]
            self.man2woman = {}
            self.woman2man = {}
            for gender in genders:
                self.man2woman[gender[1]] = gender[0]
                self.woman2man[gender[0]] = gender[1]
        with open('gender_data/{}_test.txt'.format(self.dataset), 'r', encoding='utf-8') as rf:
            datas = [each.strip().split('\t') for each in rf.readlines()]
            for data in datas:
                gender = ''
                words = data[1].split()
                for word in words:
                    if word in self.man2woman.keys():
                        gender = 'man'
                        break
                    if word in self.woman2man.keys():
                        gender = 'woman'
                        break
                data.append(gender)
                self.datas.append(data)

    def _padding(self, max_token_len, texts, pad_token):
        resutls = []
        for text in texts:
            text = text[:self.cfg.max_len]
            if len(text) < max_token_len:
                text = text + [pad_token]*(max_token_len-len(text))
            else:
                text = text[:max_token_len]
            resutls.append(text)
        return resutls

    def get_batch(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.cfg.batch:
                break
        if len(batch_data) < 1:
            return None
        # base_texts is the x' of IntegratedGrads Method, all pad_tokens
        texts, labels, disturbed_texts, masks, genders  = [], [], [], [], []
        for data in batch_data:
            label, text, disturbed_text, gender = int(data[0]), data[1], data[2], data[3]
            tokens = self.tokenizer.tokenize(text)[:self.cfg.max_len]
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            texts.append(tokens)
            # 扰动的文本处理
            disturbed_tokens = self.tokenizer.tokenize(disturbed_text)[:self.cfg.max_len]
            disturbed_tokens = [self.tokenizer.cls_token] + disturbed_tokens + [self.tokenizer.sep_token]
            disturbed_tokens = self.tokenizer.convert_tokens_to_ids(disturbed_tokens)
            disturbed_texts.append(disturbed_tokens)
            labels.append(label)
            mask = [1]*len(tokens)
            masks.append(mask)
            genders.append(gender)
        max_token_len = max(len(each) for each in texts)
        texts = self._padding(max_token_len, texts, self.PAD_ID)
        disturbed_texts = self._padding(max_token_len, disturbed_texts, self.PAD_ID)
        masks = self._padding(max_token_len, masks, 0)
        return {
            'orgin_text': torch.LongTensor(texts),
            'disturbed_text': torch.LongTensor(disturbed_texts),
            'label': torch.LongTensor(labels),
            'attention_mask': torch.LongTensor(masks),
            'gender': genders
        }

    def __next__(self):
        if self.ipts is None:
            self.reset()
        self.ipts = self.get_batch()  # each iter get a batch
        if self.ipts is None:
            raise StopIteration
        else:
            return self.ipts


if __name__ == '__main__':
    cfg = DebiasConfig()
    model, tokenizer = load_backbone(cfg)
    # loader = DebiasDataset(cfg=cfg, tokenizer=tokenizer, train=True, target=False)
    loader = AttackDataset(cfg=cfg, tokenizer=tokenizer)
    for ipt in loader:
        # print(ipt['cad_text_0'].shape)
        # print(ipt['cad_text_1'].shape)
        # print(ipt['cad_text_2'].shape)
        # print(ipt)
        pass