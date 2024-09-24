import config
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from base_loader import GenderDataset
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

cfg = config.WeakConfig()

def PR(y_true, y_prediction):
    cnf_matrix = confusion_matrix(y_true, y_prediction)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return FPR, FNR

def test_gender(ACWG, dataset):
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
    loader = GenderDataset(cfg, tokenizer, dataset, True)
    model.eval()
    targets, preds, disturbed_preds, genders = [], [], [], []
    for step, ipt in enumerate(loader):
        input_ids = ipt['orgin_text'].cuda()
        attention_mask = ipt['attention_mask'].cuda()
        disturbed_ids = ipt['disturbed_text'].cuda()
        genders.extend(ipt['gender'])
        out = model(input_ids, attention_mask).logits
        disturbed_out = model(disturbed_ids, attention_mask).logits
        target = ipt["label"].cpu().detach().numpy()
        pred = torch.max(out, dim=-1)[1].cpu().detach().numpy()
        disturbed_pred = torch.max(disturbed_out, dim=-1)[1].cpu().detach().numpy()
        targets.extend(list(target))
        preds.extend(list(pred))
        disturbed_preds.extend(list(disturbed_pred))
    # acc = accuracy_score(targets, preds)
    disturbed_agreement = accuracy_score(disturbed_preds, preds)
    FPR_overall, FNR_overall = PR(targets, preds)
    man_preds, man_targets, woman_preds, woman_targets = [], [], [], []
    for tl, pl, gender in zip(targets, preds, genders):
        if gender == 'man':
            man_preds.append(pl)
            man_targets.append(tl)
        else:
            woman_preds.append(pl)
            woman_targets.append(tl)
    FPR_man, FNR_man = PR(man_targets, man_preds)
    FPR_woman, FNR_woman = PR(woman_targets, woman_preds)
    FPED = abs(FPR_overall-FPR_man) + abs(FPR_overall-FPR_woman)
    FNED = abs(FNR_overall-FNR_man) + abs(FNR_overall-FNR_woman)
    print('============acc on {} is {}================='.format(dataset, disturbed_agreement))
    print('============FPED is {}================='.format(FPED))
    print('============FNED is {}================='.format(FNED))

if __name__ == '__main__':
    for dataset in ['ToxicTweets']:
        for bbbb in [False, True]:
            print('==================gender test on {}================'.format(dataset))
            test_gender(bbbb, dataset)

