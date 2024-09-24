from config import WeakConfig
import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr
# from base_loader import MyDataset
from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
import random
import os
import pickle
import math
import pandas as pd

def init_logger(log_name: str = "echo", log_file='log', log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    file_handler = logging.FileHandler(log_file, encoding="utf8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def seed_everything(seed=1996):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def test_pearson(targets, preds):
    '''
    The loss of the regression task from
    <MASKER: Masked Keyword Regularization for Reliable Text Classification>
    :param targets:
    :param preds:
    :return:
    '''
    targets = np.array(targets).squeeze()
    preds = np.array(preds).squeeze()
    pearson_corr = pearsonr(preds, targets)[0]
    spearman_corr = spearmanr(preds, targets)[0]
    return (pearson_corr + spearman_corr) / 2

def get_keywords(dataset='foods', top_rate=0.1):
    '''
    :param top_rate:  选择前多少关键词进行shortcut判定
    :return:
    '''
    with open("dataset/keywords.json", 'r', encoding='utf-8') as rf:
        keywords = json.load(rf)[dataset]
        # keywords = list(keywords[dataset].keys())
        # topk = int(len(keywords)*top_rate)
        # keywords = keywords[:topk]
        # print(topk)
        words = []
        for key in keywords.keys():
            if keywords[key] > 1:
                words.append(key)
            else:
                break
        print(len(words))
    return keywords

def get_antonyms(words, path, tokenizer):
    from nltk.corpus import wordnet  # Import wordnet from the NLTK
    MASK_TOKEN = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    word2antonyms = {}
    for word in tqdm(words):
        if '#' in word:
            continue
        syns = []
        for synset in wordnet.synsets(word):  # 优先选择单词的形容词词性
            syns.append([synset, synset.name().split('.')])
        syns = sorted(syns, key=lambda a:a[1][1])
        if len(syns) > 0:
            for syn in syns:
                syn = syn[0]
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        word2antonyms[word] = lemma.antonyms()[0].name()
        else:
            word2antonyms[word] = ''
    word2antonymids = {}
    for word in word2antonyms.keys():
        key_id = tokenizer.convert_tokens_to_ids(word)
        value = word2antonyms[word]
        if value == '':
            value_id = MASK_TOKEN
        else:
            value_id = tokenizer.convert_tokens_to_ids(value)
        word2antonymids[key_id] = value_id
    with open('top_shortcuts/antonyid-bert.json', 'w') as wf:
        json.dump(word2antonymids, wf)

def to_multiCUDA(cfg, model):
    model = torch.nn.DataParallel(model, cfg.cudas)
    return model

def js_div(p_output, q_output, get_softmax=True):
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = torch.softmax(p_output, dim=-1)
        q_output = torch.softmax(q_output, dim=-1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

if __name__ == '__main__':
    pass