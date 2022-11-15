import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, Dataset
import numpy as np
from label_map import *
import json
from tqdm import tqdm

def data_helper(args, tokenizer):

    D = load_name(args.train_json)
    if args.fake_label == True:
        D += load_name1()
    train_data = data_generator(D, tokenizer, max_len=args.bert_seq_length,
                                schema=schema)

    dev_data = data_generator(load_name(args.valid_json), tokenizer, max_len=args.bert_seq_length,
                              schema=schema)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_data.collate)
    dev_loader = DataLoader(dev_data, batch_size=args.val_batch_size, shuffle=True,
                            collate_fn=dev_data.collate)


    valid_data = load_name(args.valid_json)
    return train_loader, dev_loader, valid_data

def load_name(filename):
    """{"ID": "AT0010", "text": "故障现象：车速达到45Km/h时中央门锁不能落锁。",
    "spo_list": [{"h": {"name": "中央门锁", "pos": [16, 20]}, "t": {"name": "不能落锁", "pos": [20, 24]}, "relation": "部件故障"}]}
    """
    D = []
    data = json.load(open(filename, 'r', encoding='utf-8'))
    for line in data:
        D.append({
            'text': line['text'],
            'spo_list': [
                (spo['h']['name'], tuple(spo['h']['pos']), spo['relation'], spo['t']['name'], tuple(spo['t']['pos']))
                for spo in line['spo_list']]
        })
    return D

def load_name1():
    D = []
    with open('./data/fake_label.json', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            D.append({
                'text': data['text'],
                'spo_list': [
                    (
                    spo['h']['name'], tuple(spo['h']['pos']), spo['relation'], spo['t']['name'], tuple(spo['t']['pos']))
                    for spo in data['spo_list']]
            })
    return D

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """numpy函数，将序列padding到同一长度"""
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def sequence_padding_entity(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """numpy函数，将序列padding到同一长度"""
    if length is None:
        length = np.max([np.shape(x[0])[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0][0])]

    outputs = []
    for x in inputs:
        x = x[0][slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def search(tokenizer, pattern, sequence, pos):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    h_pos = pos[0]
    n = len(pattern)
    candidate = []
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            candidate.append(i)
    if len(candidate) == 0:
        return -1
    a = []
    for i in candidate:
        s = ''.join(tokenizer.decode(sequence[1:i]).split(' '))
        le = len(s)
        a.append([abs(le - h_pos), i])
    a = sorted(a, key = lambda x : x[0])
    return a[0][1]


class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema  # spo
        self.data = [self.encoder(x) for x in data]

    def __len__(self):
        return len(self.data)

    def encoder(self, item):

        text = item['text']

        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)

        input_ids = encoder_text['input_ids']
        token_type_ids = encoder_text['token_type_ids']
        attention_mask = encoder_text['attention_mask']

        spoes = set()
        for s, s_pos, p, o, o_pos in item['spo_list']:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.schema[p]
            o = self.tokenizer.encode(o, add_special_tokens=False)

            sh = search(self.tokenizer, s, input_ids, s_pos)
            oh = search(self.tokenizer, o, input_ids, o_pos)

            if sh != -1 and oh != -1:
                spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))

        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]

        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st))  # 实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh))  # 类似于TP-Linker
            tail_labels[p].add((st, ot))

        for label in entity_labels + head_labels + tail_labels:
            if not label:
                label.add((0, 0))

        # 例如entity = [{(1, 3)}, {(4, 5), (7, 9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        return text, entity_labels, head_labels, tail_labels, \
               input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    @staticmethod
    def collate(examples):

        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []

        for item in examples:
            text, entity_labels, head_labels, tail_labels, \
            input_ids, attention_mask, token_type_ids = item

            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()  # RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()

        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, \
               batch_entity_labels, batch_head_labels, batch_tail_labels


