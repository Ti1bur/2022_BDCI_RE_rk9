import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead, BertPooler
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from transformers import AutoModel
from nezha.nezha import NeZhaModel, NeZhaConfig
from label_map import *
import numpy as np

class ERENet(nn.Module):
    def __init__(self, args):
        super(ERENet, self).__init__()
        self.mention_detect = RawGlobalPointer(hiddensize=args.hidden_size, ent_type_size=2, inner_dim=args.inner_dim)
        self.s_o_head = RawGlobalPointer(hiddensize=args.hidden_size, ent_type_size=len(schema), inner_dim=args.inner_dim, RoPE=True, tril_mask=False)
        self.s_o_tail = RawGlobalPointer(hiddensize=args.hidden_size, ent_type_size=len(schema), inner_dim=args.inner_dim, RoPE=True, tril_mask=False)
        self.nezha = False
        if 'nezha' in args.bert_dir:
            self.nezha = True
            self.encoder = NeZhaModel.from_pretrained(args.bert_dir)  # bert编码器
        else:
            self.encoder = AutoModel.from_pretrained(args.bert_dir)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.nezha == False:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)['last_hidden_state']
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        outputs = self.dropout(outputs)
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)  # [batch_size, ent_type_size, seq_len, seq_len]
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs

class RDropLoss(nn.Module):
    def __init__(self, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha
        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, *args):
        assert len(args) in {2, 3}, 'RDropLoss only support 2 or 3 input args'
        # y_pred是1个Tensor
        if len(args) == 2:
            y_pred, y_true = args
            loss_sup = self.loss_sup(y_pred, y_true)  # 两个都算

            if self.rank == 'adjacent':
                y_pred1 = y_pred[1::2]
                y_pred2 = y_pred[::2]
            elif self.rank == 'updown':
                half_btz = y_true.shape[0] // 2
                y_pred1 = y_pred[:half_btz]
                y_pred2 = y_pred[half_btz:]
        # y_pred是两个tensor
        else:
            y_pred1, y_pred2, y_true = args
            loss_sup = self.loss_sup(y_pred1, y_true)

        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))
        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) *param.data +self.decay *self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.75, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None) and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None) and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    '''
    稀疏多标签交叉熵损失的torch实现
    '''
    shape = y_pred.shape  # [batch_size, ent_type_size, seq_len, seq_len]
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + 1e12
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = torch.mean(torch.sum(pos_loss + neg_loss))
    return loss

# def batch_gather(input: Tensor, indices: Tensor):
#     """
#     Args:
#         input: label tensor with shape [batch_size, n, L] or [batch_size, L]
#         indices: predict tensor with shape [batch_size, m, l] or [batch_size, l]
#     Return:
#         Note that when second dimention n != m, there will be a reshape operation to gather all value along this dimention of input 
#         if m == n, the return shape is [batch_size, m, l]
#         if m != n, the return shape is [batch_size, n, l*m]
#     """
#     if indices.dtype != torch.int64:
#         indices = torch.tensor(indices, dtype=torch.int64)
#     results = []
#     for data, indice in zip(input, indices):
#         if len(indice) < len(data):
#             indice = indice.reshape(-1)
#             results.append(data[..., indice])
#         else:
#             indice_dim = indice.ndim
#             results.append(torch.gather(data, dim=indice_dim-1, index=indice))
#     return torch.stack(results)

# def sparse_multilabel_categorical_crossentropy(label: Tensor, pred: Tensor, mask_zero=False, reduction='none'):
#      """Sparse Multilabel Categorical CrossEntropy
#          Reference: https://kexue.fm/archives/8888, https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272
#     Args:
#          label: label tensor with shape [batch_size, n, num_positive] or [Batch_size, num_positive]
#              should contain the indexes of the positive rather than a ont-hot vector
#          pred: logits tensor with shape [batch_size, m, num_classes] or [batch_size, num_classes], don't use acivation.
#          mask_zero: if label is used zero padding to align, please specify make_zero=True.
#              when mask_zero = True, make sure the label start with 1 to num_classes, before zero padding.
#      """
#     zeros = torch.zeros_like(pred[..., :1])
#     pred = torch.cat([pred, zeros], dim=-1)
#     if mask_zero:
#         infs = torch.ones_like(zeros) * float('inf')
#         pred = torch.cat([infs, pred[..., 1:]], dim=-1)
#     pos_2 = batch_gather(pred, label)
#     pos_1 = torch.cat([pos_2, zeros], dim=-1)
#     if mask_zero:
#         pred = torch.cat([-infs, pred[..., 1:]], dim=-1)
#         pos_2 = batch_gather(pred, label)
#     pos_loss = torch.logsumexp(-pos_1, dim=-1)
#     all_loss = torch.logsumexp(pred, dim=-1)
#     aux_loss = torch.logsumexp(pos_2, dim=-1) - all_loss
#     aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-16, 1)
#     neg_loss = all_loss + torch.log(aux_loss)
#     loss = pos_loss + neg_loss
#     if reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()
#     elif reduction == 'none':
#         return loss
#     else:
#         raise Exception('Unexpected reduction {}'.format(self.reduction))



class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim=64, RoPE=True, tril_mask=True):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super(RawGlobalPointer, self).__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # [seq_len, 1]

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)  # [output_dim//2]
        indices = torch.pow(10000, -2 * indices / output_dim)  # 做幂相乘 [output_dim/2]
        embeddings = position_ids * indices  # [seq_len, out_dim/2]，每一行内容是position_ids值乘indices的值
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq_len, out_dim/2, 2]
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))  # [batch_size, seq_len, out_dim/2, 2]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim)) # [batch_size, seq_len, out_dim]
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, last_hidden_state, attention_mask):
        self.device = attention_mask.device
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        outputs = self.dense(last_hidden_state)  # [batch_size, seq_len, ent_type_size * inner_dim * 2]
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)  # 将outputs在最后一个维度上切分，每一份分别是[batch_size, seq_len, inner_dim*2]，即有ent_type_size份
        outputs = torch.stack(outputs, dim=-2)  # [batch_size, seq_len, ent_type_size, inner_dim*2]
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # qw -->> kw -->> [batch_size, seq_len, ent_type_size, inner_dim]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)  # 按dim的维度重复2次
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)  # [batch_size, seq_len, inner_dim]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)  # [batch_size, seq_len, ent_type_size, inner_dim//2, 2]
            qw2 = qw2.reshape(qw.shape)  # [batch_size, seq_len, ent_type_size, inner_dim]
            qw = qw * cos_pos + qw2 * sin_pos  # [batch_size, seq_len, ent_type_size, inner_dim]
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos  # [batch_size, seq_len, ent_type_size, inner_dim]
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask  # [batch_size, ent_type_size, seq_len, seq_len]
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12   # padding部分置为负无穷
        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)  # 下三角（不包括斜对角）
            logits = logits - mask * 1e12  # 下三角部分置为负无穷

        return logits / self.inner_dim ** 0.5