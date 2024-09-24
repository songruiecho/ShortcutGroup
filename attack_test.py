'''
attack our datasset from TextAttack
'''

# STEP 1, train a TextAttack style model

import datasets
import config
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from base_loader import AttackDataset
import torch
from sklearn.metrics import accuracy_score

cfg = config.WeakConfig()

def test_attack(ACWG, attacker, max_att_tokens, dataset):
    if ACWG:
        model_dir_name = '{}_on_{}_shortcutMLM_CL_{}_{}'.format(cfg.base_model, dataset, cfg.lambda1, cfg.lambda2)
        save_path = 'save_models/cl_classifier/' + model_dir_name
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
        model = AutoModelForSequenceClassification.from_pretrained(save_path).cuda()
    else:
        model_dir_name = '{}_on_{}'.format(cfg.base_model, dataset)
        save_path = 'save_models/weakclassifier/' + model_dir_name
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)
        model = AutoModelForSequenceClassification.from_pretrained(save_path).cuda()
    loader = AttackDataset(cfg, tokenizer, max_attack_tokens=max_att_tokens, attacker=attacker, dataset=dataset)
    model.eval()
    targets, preds = [], []
    for step, ipt in enumerate(loader):
        input_ids = ipt['orgin_text'].cuda()
        attention_mask = ipt['attention_mask'].cuda()
        out = model(input_ids, attention_mask).logits
        target = ipt["label"].cpu().detach().numpy()
        pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
        targets.extend(list(target))
        preds.extend(list(pred))
    acc = accuracy_score(targets, preds)
    print('============acc on {} is {}================='.format(dataset, acc))

for bbbb in [False]:
    for dataset in ['sst2']:
        for attacker in ['pwws', 'textbugger', 'textfooler']:
            for max_att_tokens in [1,2,3]:
                print('==================attack on {} with {} for times {}================'.format(dataset, attacker, max_att_tokens))
                test_attack(bbbb, attacker, max_att_tokens, dataset)
