import torch
import json
import sys
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import configparser
from model import RawGlobalPointer, sparse_multilabel_categorical_crossentropy, ERENet
from data_helper import data_generator, load_name
from transformers import BertModel, BertTokenizerFast
from config import *
from label_map import *
args = parse_args()
import torch.nn as nn
tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir, do_lower_case=True)

logits1 = np.load('./logits/nezha_zh_fold1.npy', allow_pickle=True)
logits2 = np.load('./logits/nezha_zh_fold2.npy', allow_pickle=True)
logits3 = np.load('./logits/nezha_zh_fold3.npy', allow_pickle=True)
logits4 = np.load('./logits/nezha_zh_fold4.npy', allow_pickle=True)
logits5 = np.load('./logits/nezha_zh_fold5.npy', allow_pickle=True)
logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
logits1 = np.load('./logits/nezha_wwm_fold1.npy',allow_pickle=True)
logits2 = np.load('./logits/nezha_wwm_fold2.npy',allow_pickle=True)
logits3 = np.load('./logits/nezha_wwm_fold3.npy',allow_pickle=True)
logits4 = np.load('./logits/nezha_wwm_fold4.npy',allow_pickle=True)
logits5 = np.load('./logits/nezha_wwm_fold5.npy',allow_pickle=True)
logits += (logits1 + logits2 + logits3 + logits4 + logits5) / 5

logits1 = np.load('./logits/roberta_fold1.npy',allow_pickle=True)
logits2 = np.load('./logits/roberta_fold2.npy',allow_pickle=True)
logits3 = np.load('./logits/roberta_fold3.npy',allow_pickle=True)
logits4 = np.load('./logits/roberta_fold4.npy',allow_pickle=True)
logits5 = np.load('./logits/roberta_fold5.npy',allow_pickle=True)
logits += (logits1 + logits2 + logits3 + logits4 + logits5) / 5

logits1 = np.load('./logits/macbert_fold1.npy',allow_pickle=True)
logits2 = np.load('./logits/macbert_fold2.npy',allow_pickle=True)
logits3 = np.load('./logits/macbert_fold3.npy',allow_pickle=True)
logits4 = np.load('./logits/macbert_fold4.npy',allow_pickle=True)
logits5 = np.load('./logits/macbert_fold5.npy',allow_pickle=True)
logits += (logits1 + logits2 + logits3 + logits4 + logits5) / 5
logits /= 4
# path = [f'./fold{i}.npy' for i in range(1, 4)]
# for i in range(len(path)):
#     if i  == 0:
#         logits = np.load(path[i], allow_pickle = True)
#     else:
#         logits += np.load(path[i], allow_pickle = True)
data = []
with open('./data/test_baseline.json', 'r', encoding='utf-8') as f:
    text_list = [(x['text'], x['id']) for x in json.load(f)]
    idx = -1
    for text, id_ in tqdm(text_list):
        idx += 1
        result = {}
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.bert_seq_length)['offset_mapping']
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        threshold = 0.0
        outputs = logits[idx]
        subjects, objects = set(), set()
        outputs[0][:, [0, -1]] -= np.inf  # 首尾取负无穷
        outputs[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs[0] > 0)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))
        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add((
                        text[new_span[sh][0]:new_span[st][-1] + 1], (new_span[sh][0], new_span[st][-1] + 1), id2schema[p],
                        text[new_span[oh][0]:new_span[ot][-1] + 1], (new_span[oh][0], new_span[ot][-1] + 1)
                    ))
        spo_list = []
        result['ID'] = id_
        result['text'] = text
        for spo in list(spoes):
            spo_list.append({'h': {'name': spo[0], 'pos': list(spo[1])}, 't': {'name': spo[3], 'pos': list(spo[4])}, 'relation': spo[2]})
        result["spo_list"] = spo_list
        data.append(json.dumps(result, ensure_ascii=False))

    with open('./data/result.json', 'w', encoding='utf-8') as w:
        for line in data:
            w.write(line)
            w.write('\n')



# param_path = [0] + ['fold1_epoch18_f1_0.7121118825454591', 'fold2_epoch20_f1_0.7029426463740862', 'fold3_epoch15_f1_0.6823611439195026', 'fold4_epoch12_f1_0.6871269027036592','fold5_epoch14_f1_0.6978106353284647']
# for fold in range(1, 6):
#     model = ERENet(args).to('cuda')
#     model.load_state_dict(torch.load('./save/fold5/nezha-wwm/'+param_path[fold]+'.bin', map_location='cpu'))
#     model.eval()
#     logits = []
#     with open('./data/test_baseline.json', 'r', encoding='utf-8') as f:
#         text_list = [(x['text'], x['id']) for x in json.load(f)]
#         idx = -1
#         for text, id_ in tqdm(text_list):
#             idx += 1
#             result = {}
#             token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.bert_seq_length)['offset_mapping']
#             new_span, entities = [], []
#             for i in token2char_span_mapping:
#                 if i[0] == i[1]:
#                     new_span.append([])
#                 else:
#                     if i[0] + 1 == i[1]:
#                         new_span.append([i[0]])
#                     else:
#                         new_span.append([i[0], i[-1] - 1])
#             threshold = 0.0
#             encoder_txt = tokenizer.encode_plus(text, max_length=args.bert_seq_length)
#             input_ids = torch.tensor(encoder_txt['input_ids']).long().unsqueeze(0).to('cuda')
#             token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to('cuda')
#             attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to('cuda')
#             scores = model(input_ids, attention_mask, token_type_ids)
#             outputs = [o[0].data.cpu().numpy() for o in scores]
#             logits.append(outputs)
#     np.save(f'./logits/nezha_wwm_fold{fold}.npy', np.array(logits))

