import re
from tqdm import tqdm

with open('../dataset/ToxicTweets_test.txt', 'r', encoding='utf-8') as rf:
    Toxic = [each.strip().split('\t') for each in rf.readlines()]

with open('../dataset/Davids_test.txt', 'r', encoding='utf-8') as rf:
    Davidson = [each.strip().split('\t') for each in rf.readlines()]

with open('gender.txt', 'r', encoding='utf-8') as rf:
    genders = [each.strip().split('-') for each in rf.readlines()]
    man2woman = {}
    woman2man = {}
    for gender in genders:
        man2woman[gender[1]] = gender[0]
        woman2man[gender[0]] = gender[1]
    genders = list(man2woman.keys()) + list(woman2man.keys())

# 读取性别之后进行包含性别的单词的查找
GenderToxic, GenderDavidson = [], []
for each in tqdm(Toxic):
    disturbed_text = []
    has_gender = False
    try:
        words = each[1].split()
    except:
        continue
    for word in words:
        if word in genders:
            has_gender = True
            if word in man2woman.keys():
                disturbed_text.append(man2woman[word])
            else:
                disturbed_text.append(woman2man[word])
        else:
            disturbed_text.append(word)
    if has_gender:
        GenderToxic.append('\t'.join([each[0], each[1], ' '.join(disturbed_text)]))

for each in tqdm(Davidson):
    disturbed_text = []
    has_gender = False
    try:
        words = each[1].split()
    except:
        continue
    for word in words:
        if word in genders:
            has_gender = True
            if word in man2woman.keys():
                disturbed_text.append(man2woman[word])
            else:
                disturbed_text.append(woman2man[word])
        else:
            disturbed_text.append(word)
    if has_gender:
        GenderDavidson.append('\t'.join([each[0], each[1], ' '.join(disturbed_text)]))

print(len(GenderToxic))
print(len(GenderDavidson))

with open('ToxicTweets_test.txt', 'w', encoding='utf-8') as wf:
    wf.write('\n'.join(GenderToxic))

with open('Davids_test.txt', 'w', encoding='utf-8') as wf:
    wf.write('\n'.join(GenderDavidson))
# 使用两个指标：性别翻转后的预测结果发生改变的比例

# 以及，针对男性和女性样本的统一的准确率