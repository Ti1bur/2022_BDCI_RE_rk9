import json
import re


def train_generator(read, save):
    fr = open(read, encoding='utf8').readlines()
    fw = open(save, 'w', encoding='utf8')

    arr_all = []

    for i in fr:
        i = i.strip()
        if i == "":
            continue

        dic_single = {}
        arr_single = []

        data = json.loads(i)
        id = data['ID']
        text = data['text']
        spo_list = data['spo_list']

        # id_input = int(id.replace('AT', '').lstrip('0'))
        dic_single['id'] = id
        dic_single['text'] = text
        dic_single['spo_list'] = []

        if text in arr_all:
            continue

        if len(text) > 300:
            for spo in spo_list:
                h = spo['h']
                t = spo['t']
                relation = spo['relation']
                line = [(h['pos'][0], h['pos'][1], h['name']), relation, (t['pos'][0], t['pos'][1], t['name'])]
                arr_single.append(line)

            # dict_all[text] = arr_single
            spos = sorted(arr_single)

            split_blocks = cut_pattern.split(text)
            split_blocks.append("")
            split_blocks = ["".join(i) for i in zip(split_blocks[0::2], split_blocks[1::2])]
            current_text = ""
            total_blocks = []
            for block in split_blocks:
                if len(current_text + block) > 300:
                    total_blocks.append(current_text)
                    current_text = block
                else:
                    current_text += block

            if len(current_text) > 0:
                total_blocks.append(current_text)

            start_idx = 0
            end_idx = 0
            for t_idx, block_text in enumerate(total_blocks):

                end_idx += len(block_text)
                new_spos = []
                for spo in spos:

                    h_sidx, h_eidx, h_name = spo[0]
                    t_sidx, t_eidx, t_name = spo[2]

                    if start_idx <= h_eidx < end_idx and start_idx <= t_eidx <= end_idx:
                        new_spos.append(spo)

                if t_idx == 0:
                    line = {"id": id, "text": block_text, "spo_list": new_spos}
                    arr_all.append(line)

                else:
                    new_spos2 = []
                    for spo in new_spos:
                        h_sidx, h_eidx, h_name = spo[0]
                        relation = spo[1]
                        t_sidx, t_eidx, t_name = spo[2]
                        tmp = []
                        tmp.append((h_sidx - start_idx, h_eidx - start_idx, h_name))
                        tmp.append(relation)
                        tmp.append((t_sidx - start_idx, t_eidx - start_idx, t_name))
                        new_spos2.append(tmp)

                    line = {"id": id, "text": block_text, "spo_list": new_spos2}
                    arr_all.append(line)
                start_idx = end_idx

        else:
            for spo in spo_list:
                h = spo['h']
                t = spo['t']
                relation = spo['relation']

                arr_h = []
                arr_h.append(h['pos'][0])
                arr_h.append(h['pos'][1])
                arr_h.append(h['name'])

                arr_t = []
                arr_t.append(t['pos'][0])
                arr_t.append(t['pos'][1])
                arr_t.append(t['name'])

                arr_spo = []
                arr_spo.append(arr_h)
                arr_spo.append(relation)
                arr_spo.append(arr_t)
                dic_single['spo_list'].append(arr_spo)

            arr_all.append(dic_single)

    fw.writelines(json.dumps(arr_all, ensure_ascii=False, indent=2))


def test_generator():
    fr = open('./data/evalB.json', 'r', encoding='utf8').readlines()
    fw = open('./data/test_baseline.json', 'w', encoding='utf8')

    datas = []
    for case in fr:
        case_data = json.loads(case)
        idx = case_data['ID']
        txt = case_data['text']
        if len(txt) > 300:
            split_blocks = cut_pattern.split(txt)
            split_blocks.append("")

            split_blocks = ["".join(i) for i in zip(split_blocks[0::2], split_blocks[1::2])]
            current_text = ""
            total_blocks = []
            for block in split_blocks:
                if len(current_text + block) > 300:
                    total_blocks.append(current_text)
                    current_text = block
                else:
                    current_text += block

            if len(current_text) > 0:
                total_blocks.append(current_text)

            for sub_idx, block in enumerate(total_blocks):
                line = {"id": str(idx) + "_{}".format(sub_idx), "text": block}
                datas.append(line)
        else:
            line = {"id": str(idx), "text": txt}
            datas.append(line)

    json.dump(datas, fw, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import json
    import copy
#     ccl_data = []
#     with open('./data/ccl2022.json', 'r', encoding='utf8') as f:
#         lines = f.readlines()
#         for line in lines:
#             ccl_data.append(json.loads(line))
#     train_data = []
#     with open('./data/train.json', 'r', encoding='utf8') as f:
#         lines = f.readlines()
#         for line in lines:
#             train_data.append(json.loads(line))
#     new_data = []
#     dit = {}
#     for i in ccl_data:
#         text = i['text']
#         if text in dit.keys():
#             dit[text]['spo_list'] += i['spo_list']
#         else:
#             dit[text] = {'spo_list': i['spo_list'], 'ID': i['ID']}
#     for k, v in dit.items():
#         new_data.append({'ID': v['ID'], 'text': k, 'spo_list': v['spo_list']})
#     ccl_data = copy.deepcopy(new_data)
#     a = 1
#     def merge(s_text, s_spo, l_text, l_spo):
#         text = l_text
#         spo = l_spo
#         idx = 0
#         for i in range(len(l_text)):
#             if l_text[i:i + len(s_text)] == s_text:
#                 idx = i
#                 break
#         for j in range(len(s_spo)):
#             s_spo[j]['h']['pos'][0] += idx
#             s_spo[j]['h']['pos'][1] += idx
#             s_spo[j]['t']['pos'][0] += idx
#             s_spo[j]['t']['pos'][1] += idx
#         for j in s_spo:
#             if j not in spo:
#                 spo.append(j)
#         return text, spo


#     #外部数据中存在文本与训练集以及A榜测试集相互包含的清空，所以为了方式交叉验证的时候数据泄露，所以对外部数据集进行清洗
# #     for train_idx, train_sample in enumerate(train_data):
# #         for outer_idx, outer_sample in enumerate(ccl_data):
# #             # 训练集文本被包含在外部数据的文本中，则将外部数据加入训练集，并将其从外部数据集中剔除
# #             if train_sample['text'] in outer_sample['text']:
# #                 text, spo = merge(train_sample['text'], train_sample['spo_list'], outer_sample['text'],
# #                                   outer_sample['spo_list'])
# #                 ccl_data.pop(outer_idx)
# #                 train_data[train_idx] = {
# #                     "ID": train_sample["ID"], "text": text, "spo_list": spo
# #                 }
# #             # 外部数据的文本被包含在训练集文本中，则将外部数据剔除防止穿越
# #             elif outer_sample['text'] in train_sample['text']:
# #                 text, spo = merge(outer_sample['text'], outer_sample['spo_list'], train_sample['text'],
# #                                   train_sample['spo_list'])
# #                 ccl_data.pop(outer_idx)
# #                 train_data[train_idx] = {
# #                     "ID": train_sample["ID"], "text": text, "spo_list": spo
# #                 }
#     for train_idx, train_sample in enumerate(train_data):
#         for outer_idx, outer_sample in enumerate(ccl_data):
#             # 训练集文本被包含在外部数据的文本中，则将外部数据加入训练集，并将其从外部数据集中剔除
#             if train_sample['text'] in outer_sample['text']:
#                 ccl_data.pop(outer_idx)
#             elif outer_sample['text'] in train_sample['text']:
#                 ccl_data.pop(outer_idx)
    
    
#     with open('./data/new_train_ccl2022.json', 'w', encoding='utf8') as f:
#         for line in ccl_data:
#             f.write(json.dumps(line, ensure_ascii=False))
#             f.write('\n')
#     with open('./data/new_train_data.json', 'w', encoding='utf8') as f:
#         for line in train_data:
#             f.write(json.dumps(line, ensure_ascii=False))
#             f.write('\n')
    cut_pattern = re.compile(r'([，。！？、])')
#     train_generator('./data/new_train_data.json', './data/train_baseline.json')
    test_generator()
    
    
#     ccl_data = []
#     with open('./data/new_train_baseline.json', 'r', encoding='utf8') as f:
#         ccl_data = json.load(f)
#     test_data = []
#     with open('./data/test_baseline.json', 'r', encoding='utf8') as f:
#         test_data = json.load(f)
#     s = set()
#     new_data = []
#     for i in test_data:
#         s.add(i['text'])
#     for data in ccl_data:
#         ok = 1
#         for j in s:
#             if data['text'] in j or j in data['text']:
#                 ok = 0
#         if ok == 1:
#             new_data.append(data)
#     with open('./data/new_train_baseline_without_test.json', 'w', encoding='utf8') as f:
#         json.dump(new_data, f, ensure_ascii=False, indent=4)
#     D = []
#     with open('./data/new_train_baseline_with_test.json', 'w', encoding='utf-8') as f:
#         for line in new_data:
#             new_line = {}
#             new_line['ID'] = line['id']
#             new_line['text'] = line['text']
#             spo_list = []
#             for j in line['spo_list']:
#                 spo = {}
#                 spo['h'] = {'name': j[0][2], 'pos': [j[0][0], j[0][1]]}
#                 spo['t'] = {'name': j[2][2], 'pos': [j[2][0], j[2][1]]}
#                 spo['relation'] = j[1]
#                 spo_list.append(spo)
#             new_line['spo_list'] = spo_list
#             D.append(new_line)
#         json.dump(D, f, ensure_ascii=False, indent=4)