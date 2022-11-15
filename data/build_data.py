# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/9/4 10:54
# software: PyCharm

"""
文件说明：
    mean: 172, 75% 235
    总共1491条数据
"""
import json
import pandas as pd
import random

seed = 2022
random.seed(seed)
t = set()
text = []
D = []
with open('train_baseline.json', 'r', encoding='utf-8') as f:
    file = json.load(f)
    for line in file:
        new_line = {}
        new_line['id'] = line['id']
        new_line['text'] = line['text']
        spo_list = []
        for j in line['spos']:
            spo = {}
            spo['h'] = {'name' : j[0][2], 'pos':[j[0][0], j[0][1]]}
            spo['t'] = {'name' : j[2][2], 'pos':[j[2][0], j[2][1]]}
            spo['relation'] = j[1]
            spo_list.append(spo)
        new_line['spo_list'] = spo_list
        D.append(new_line)

print(len(D))
random.shuffle(D)
le = len(D) // 5
a, b, c, d, e = D[:le], D[le:2*le], D[2*le:3*le], D[3*le:4*le], D[4*le:]
x = [a, b, c, d, e]
for i in range(5):
    data = []
    for j in range(5):
        if i != j:
            data += x[j]
        else:
            dev = x[j]
    with open(f'./{i + 1}train.json','w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    with open(f'./{i + 1}valid.json','w', encoding='utf8') as f:
        json.dump(dev, f, ensure_ascii=False, indent=4)

#         spos = line['spo_list']
#         text.append(line['text'])
#         for spo in spos:
#             t.add(spo['relation'])
# print(t)
# df = pd.DataFrame({'text': text})
# print(df.text.apply(len).describe())
