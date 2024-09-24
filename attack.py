import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import pandas as pd
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import PWWSRen2019, TextBuggerLi2018, TextFoolerJin2019
from textattack.transformations import WordSwapEmbedding
import textattack
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config
from textattack.loggers import CSVLogger
import torch
from textattack.constraints.overlap import LevenshteinEditDistance, MaxWordsPerturbed
from textattack.search_methods import BeamSearch

def attack(dataset_name, attacker_name):
    # 1. 准备数据集
    datas = []
    if dataset_name == 'foods':
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
    if dataset_name == 'sst2':
        file = open('dataset/sst2_dev.tsv', 'r', encoding='utf-8')
        for line in file.readlines():
            line = line.strip().split('\t')
            datas.append([line[0].lower(), int(line[1])])

    if dataset_name in ['mr', 'imdb_s', 'kindle', 'amazon']:
        file = open('dataset/{}_test.txt'.format(dataset_name), 'r', encoding='utf-8')
        lists = [each for each in file.readlines()]
        for line in lists:
            line = line.strip().split()
            datas.append([' '.join(line[1:]), int(line[0])])
    if dataset_name in ['Davids', 'HatEval', 'OffEval', 'Abusive', 'StormW', 'ToxicTweets',
                        'books', 'dvd', 'electronics', 'kitchen']:
        file = open('dataset/{}_test.txt'.format(dataset_name), 'r', encoding='utf-8')
        lists = [each for each in file.readlines()]
        for line in lists:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            datas.append([line[1], int(line[0])])

    dataset = textattack.datasets.Dataset(datas)

    # 2. 定义模型
    root = '/home/songrui/codes/ShortcutGroup/save_models/weakclassifier/'

    model = AutoModelForSequenceClassification.from_pretrained(root+'bert_on_{}'.format(dataset_name)) # replace this line with your model loading code
    # model = torch.nn.DataParallel(model, [0,1,2,3])
    tokenizer = AutoTokenizer.from_pretrained('/home/songrui/datas/bert-base-uncased/')

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model.cuda(), tokenizer)
    attack_args = textattack.AttackArgs(num_examples=len(datas), parallel=True)

    # 3. 定义转换器和约束条件
    # transformation = WordSwapEmbedding()
    # constraint = WordEmbeddingDistance(min_cos_sim=0.5)

    # 4. 运行攻击
    if attacker_name == 'textfooler':
        attack = TextFoolerJin2019.build(model_wrapper)
    if attacker_name == 'pwss':
        attack = PWWSRen2019.build(model_wrapper)
    if attacker_name == 'textbugger':
        attack = TextBuggerLi2018.build(model_wrapper)
    # 添加约束
    # attack.search_method = BeamSearch(beam_width=2)
    # attack.constraints.append(MaxWordsPerturbed(3))
    attacker = textattack.Attacker(attack, dataset, attack_args)
    results_iterable = attacker.attack_dataset()

    # 5. 写入文件
    logger = CSVLogger(filename='attack_data/{}_attack_{}.csv'.format(attacker_name, dataset_name))
    for result in results_iterable:
        logger.log_attack_result(result)

    logger.flush()

if __name__ == '__main__':
    # 不对这些长文本攻击了，太费劲（超过24h在4*A40上运行）
    for dataset in ['books', 'dvd', 'electronics', 'kitchen']:
        for attacker in ['textfooler']:
            attack(dataset, attacker)