import json

with open('./data/result.json', 'r', encoding='utf8') as f:
    data = [json.loads(x) for x in f.readlines()]
    test = []
    for x in data:
        if '_' in x['ID']:
            test.append((x['ID'][:6], int(x['ID'][-1]), x['text'], x['spo_list']))
        else:
            test.append((x['ID'], 0, x['text'], x['spo_list']))
test = sorted(test)
dict = {}
for i in test:
    id, text, spo = i[0], i[2], i[3]
    if id in dict.keys():
        le = len(dict[id]['text'])
        for j in spo:
            tmp = j
            tmp['h']['pos'][0] += le
            tmp['h']['pos'][1] += le
            tmp['t']['pos'][0] += le
            tmp['t']['pos'][1] += le
            dict[id]['spo_list'].append(tmp)
        dict[id]['text'] += text
    else:
        dict[id] = {'text':text, 'spo_list': spo}
data = []
for id in dict.keys():
    tmp = {}
    tmp['ID'] = id
    tmp['text'] = dict[id]['text']
    tmp['spo_list'] = dict[id]['spo_list']
    data.append(json.dumps(tmp, ensure_ascii=False))
with open('./data/merge_result.json', 'w', encoding='utf8') as f:
    for line in data:
        f.write(line)
        f.write('\n')
