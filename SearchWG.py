'''
Search Word-group by greedy search algorithm
'''
import os
import traceback
import json
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

from config import WeakConfig, DebiasConfig
import torch.optim as optim
from Classifier import WeakClassifier, DebiasClassifier
from base_loader import MyDataset, DebiasDataset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from utils import *
import torch
import torch.nn as nn
from utils import *

MAX_WG_LENGTH = 3    # 设置最大的词组长度为3
BEEM_WIDTH = 2       # 束搜索的宽度为3
with open('antony.json', 'rb') as rf:
    ANTONY = json.load(rf)

# 加载已经训练好的模型
weak_cfg = WeakConfig()
weak_cfg.use_checkpoints = True
weak_cfg.batch = 1    # 这里需要是1
attn_model = WeakClassifier(weak_cfg)
tokenizer = attn_model.tokenizer
train_loader = MyDataset(weak_cfg, tokenizer, train=True, target=False, shuffle=False)
test_loader = MyDataset(weak_cfg, tokenizer, train=False, target=False, shuffle=False)
train_texts = train_loader.datas
test_texts = test_loader.datas
attn_model = attn_model.model.cuda()   # BertForSequenceClassification

def disturbe_text(cand_word, text):  # 给出一个单词，对包含单词的本文进行扰动
    '''
    :param words:
    :param text:
    :return:
    '''
    disturbed_text = []
    if '#' in text:
        words = text.split()
    else:
        words = tokenizer.tokenize(text)
    if cand_word in ANTONY.keys():
        replace_word = ANTONY[cand_word]  # 被替换的单词
    else:
        replace_word = '[MASK]'
    replace_word = tokenizer.tokenize(replace_word)
    for w in words:
        if w == cand_word:   # 候选词替换
            if len(replace_word) == 0:
                disturbed_text.append('[MASK]')
            else:
                disturbed_text.extend(replace_word)
        else:
            disturbed_text.append(w)
    return ' '.join(disturbed_text)

def cal_jsd(base_logits, disturbed_texts, disturbed_words):
    ''' 根据未添加扰动的logit以及扰动之后的disturbed_texts返回相应的扰动样本的因果得分
    :param base_logits:
    :param disturbed_texts:
    :return:
    '''
    disturbed_ipts = tokenizer(disturbed_texts, padding=True, return_tensors="pt")  # 前向传播计算jsd的输入数据
    disturbed_input = disturbed_ipts['input_ids'].cuda()
    disturbed_mask = disturbed_ipts['attention_mask'].cuda()
    disturbed_logits = attn_model(disturbed_input, disturbed_mask).logits
    causal_scores = []
    for b in range(disturbed_logits.shape[0]):
        disturbed_logit = disturbed_logits[b]
        causal_scores.append(([disturbed_words[b]],
                              float(js_div(base_logits, disturbed_logit).detach().cpu().numpy())))
    causal_scores = sorted(causal_scores, key=lambda a: a[1], reverse=True)[:BEEM_WIDTH]
    return causal_scores

def write_WG(WordGroups, write_path=''):
    ''' write Word Group to file
    :param write_path:
    :return: 没有返回值，给我滚
    '''
    results = []
    for WordGroup in WordGroups:
        wg_json = {}
        for Group in WordGroup:
            key = ' '.join(Group[0])
            value = Group[1]
            wg_json[key] = value
        results.append(wg_json)
    with open(write_path, 'w') as f:
        json.dump(results, f)

# 加载候选词
# with open('top_att_for_each_sample/train-{}.txt'.format(weak_cfg.dataset), 'r', encoding='utf-8') as rf:
#     train_causal_cands = [each.strip().split() for each in rf.readlines()]
# with open('top_att_for_each_sample/test-{}.txt'.format(weak_cfg.dataset), 'r', encoding='utf-8') as rf:
#     test_causal_cands = [each.strip().split() for each in rf.readlines()]

# with open('top_ig_for_each_sample/train-{}.txt'.format(weak_cfg.dataset), 'r', encoding='utf-8') as rf:
#     train_causal_cands = [each.strip().split() for each in rf.readlines()]
# with open('top_ig_for_each_sample/test-{}.txt'.format(weak_cfg.dataset), 'r', encoding='utf-8') as rf:
#     test_causal_cands = [each.strip().split() for each in rf.readlines()]

with open('global_ig_word_group_for_each_sample/train-{}.txt'.format(weak_cfg.dataset), 'r', encoding='utf-8') as rf:
    train_causal_cands = [each.strip().split() for each in rf.readlines()]
with open('global_ig_word_group_for_each_sample/test-{}.txt'.format(weak_cfg.dataset), 'r', encoding='utf-8') as rf:
    test_causal_cands = [each.strip().split() for each in rf.readlines()]

# Loaders = {'train':train_loader, 'test':test_loader}
Loaders = {'test':test_loader}
for key in Loaders.keys():
    print('============{}============='.format(key))
    Final_Group_Words = []
    loader = Loaders[key]
    for step, ipt in tqdm(enumerate(loader)):
        # if step == 10:   # 测试用的，别管
        #     break
        # Beam Search
        # 首先，生成一个beam宽度的列表，存放topK个因果的词 √
            # 具体的选择过程：依次生成扰动的样本；扰动的样本作为一个batch计算logits；计算相应的散度，并选择topK进行更新
        # 当前循环中从中选出和循环数量长度一样的候选group，并在此基础上进行继续的扩充
        tokens = ipt['orgin_text'].cuda()
        masks = ipt['attention_mask'].cuda()
        with torch.no_grad():
            base_logits = attn_model(tokens, masks, output_attentions=True).logits.squeeze()
        if key == 'train':
            text = train_texts[step]
            try:
                cand_words = train_causal_cands[step]
            except:
                cand_words = []
        else:
            text = test_texts[step]
            try:
                cand_words = test_causal_cands[step]
            except:
                cand_words = []
        cand_words = list(set(cand_words))
        Word_Groups = []    # 存放搜索出来的词组, 数组类型为[[w1,w2,..],score]
        words = tokenizer.tokenize(text[0])
        # beam search start
        # 结束条件：直到搜索出来的word-group长度达到一定的长度，或者beem中的每一个候选group新的增长都没有办法造成JSD的增长，则停止搜索
        for i in range(min(len(words), MAX_WG_LENGTH)):    # max words, 注意有的时候句子很短，比最大的搜索长度都短，因此需要min添加比较
            disturbed_texts = []  # 扰动之后的样本
            disturbed_words = []  # 扰动样本对应的扰动词
            if i == 0:        # 首先计算top-width最大因果效应的单个词
                try:
                    concurrent_cand_words = cand_words
                    for cand_word in concurrent_cand_words:  # 每个候选词都需要进行扰动
                        disturbed_text = disturbe_text(cand_word, text[0]) # 添加扰动
                        disturbed_words.append(cand_word)
                        disturbed_texts.append(disturbed_text)
                    # 统一计算扰动样本们的散度得分（因果效应）
                    causal_scores = cal_jsd(base_logits, disturbed_texts, disturbed_words)
                    Word_Groups.extend(causal_scores)  # 追加
                except:
                    print(text[0])
                    continue
            else:
                # 先对已有的词组里面的长度为step的数据进行检索，因为要在他们的基础上继续增长
                cand_word_groups = []  # 存放本轮筛出的候选的word-group
                for group in Word_Groups:
                    current_group_words = group[0]
                    disturbed_texts = []  # 扰动之后的样本
                    disturbed_words = []  # 扰动样本对应的扰动词
                    if len(current_group_words) == i:
                        try:
                            concurrent_cand_words = set(cand_words) - set(current_group_words)  # 当前需要检索的候选词表
                            current_text = text[0]
                            for current_group_word in current_group_words:
                                # 已经被扰动过的样本作为base样本，需要在其基础上继续添加扰动
                                current_text = disturbe_text(current_group_word, current_text)
                            current_base_text = current_text
                            for concurrent_cand_word in concurrent_cand_words:
                                disturbed_words.append(concurrent_cand_word)
                                disturbed_text = disturbe_text(concurrent_cand_word, current_base_text)
                                disturbed_texts.append(disturbed_text)
                            causal_scores = cal_jsd(base_logits, disturbed_texts, disturbed_words)
                            for score in causal_scores:  # 去重复保存新搜索到的词
                                new_group_words = sorted(current_group_words + score[0])
                                new_group_words = (new_group_words, score[1])
                                if new_group_words not in cand_word_groups:
                                    cand_word_groups.append(new_group_words)
                        except:
                            # print(current_text)
                            continue
                cand_word_groups = sorted(cand_word_groups, key=lambda a:a[1], reverse=True)[:BEEM_WIDTH]
                Word_Groups.extend(cand_word_groups)
        Word_Groups = sorted(Word_Groups, key=lambda a:a[1], reverse=True)
        Final_Group_Words.append(Word_Groups)
    path = 'global_ig_word_group_for_each_sample/{}-{}.json'.format(key, weak_cfg.dataset)
    write_WG(Final_Group_Words, path)