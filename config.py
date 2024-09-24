import torch

class BaseConfig:
    def __init__(self):
        super(BaseConfig, self).__init__()
        self.get_attributes()

    def get_attributes(self):
        self.base_model = 'roberta'  # Bert, DistilBert, Alberta, RoBERTa
        self.root = '/home/songrui/datas/'   # 需要改成你自己的路径
        self.use_checkpoints = False  # 是否使用已经训练好的节点
        if self.base_model == 'bert':
            self.bert_path = self.root + 'bert-base-uncased/'
        if self.base_model == 'distilbert':
            self.bert_path = self.root + 'distillbert_en_base/'
        if self.base_model == 'roberta':
            self.bert_path = self.root + 'roberta_en_base/'
        if self.base_model == 'alberta':
            self.bert_path = self.root + 'albert_base_v2/'
        self.dataset = 'kindle'
        self.target = 'foods'
        if self.dataset in ['Davids', 'OffEval', 'Abusive', 'ToxicTweets']:
            self.max_len = 50
        else:
            self.max_len = 150
        if self.dataset in ['foods', 'sst2', 'imdb_s', 'mr', 'kindle',
                            'Davids', 'OffEval', 'Abusive', 'ToxicTweets',
                            'books', 'dvd', 'electronics', 'kitchen']:
            self.nclass = 2
        if self.dataset in ['telephone', 'letters', 'facetoface']:
            self.nclass = 3
        if self.dataset == 'amazon':
            self.nclass = 50
        # set cuda
        self.cudas = [0,1]
        self.lambda1 = 0.001
        self.lambda2 = 0.001

class DebiasConfig(BaseConfig):
    def __init__(self):
        super(DebiasConfig, self).__init__()
        self.lr = 1e-5
        self.epoch = 5
        self.batch = 64
        self.get_attributes()

class WeakConfig(BaseConfig):
    '''
    weak model的配置
    '''
    def __init__(self):
        super(WeakConfig, self).__init__()
        self.lr = 1e-5
        self.epoch = 5
        self.batch = 64
        self.get_attributes()