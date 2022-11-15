from config import parse_args
from label_map import *
from utils import *
from tqdm import tqdm
from transformers import BertTokenizerFast
from data_helper import *
from model import ERENet, EMA, FGM, sparse_multilabel_categorical_crossentropy
import time
import warnings
import logging
warnings.filterwarnings("ignore")

from imp import reload
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"./log/train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)


class SPO(tuple):
    """用来存三元组的类，表现跟tuple基本一致，重写了两个特殊方法，使得在判断两个三元组是否等价时容错性更好"""
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0], add_special_tokens=False)),
            tuple(spo[1]),
            spo[2],
            tuple(tokenizer.tokenize(spo[3], add_special_tokens=False)),
            tuple(spo[4])
        )

    def __hash__(self):
        return self.spox.__hash__()
    def __eq__(self, spo):
        return self.spox == spo.spox

def extract_spoes(args, text,  threshold=0.0, model=None):
    """抽取输入text中所包含的三元组"""
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.bert_seq_length)['offset_mapping']
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:  # 单个字
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])  # 闭区间
    encoder_txt = tokenizer.encode_plus(text, max_length=args.bert_seq_length)
    input_ids = torch.tensor(encoder_txt['input_ids']).long().unsqueeze(0).to('cuda')
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to('cuda')
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to('cuda')
    scores = model(input_ids, attention_mask, token_type_ids)
    outputs = [o[0].data.cpu().numpy() for o in scores]  # list类型，每个位置形状[ent_type_size, seq_len, seq_len]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf  # 在seq_len维度首尾取负无穷
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > 0)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)  # 取交集
            for p in ps:
                spoes.add((
                    text[new_span[sh][0]:new_span[st][-1] + 1], (new_span[sh][0], new_span[st][-1] + 1), id2schema[p],
                    text[new_span[oh][0]:new_span[ot][-1] + 1], (new_span[oh][0], new_span[ot][-1] + 1)
                ))
    return list(spoes)


def evaluate(args, data, model):
    """评估函数，计算f1、Precision、Recall"""
    model.eval()
    X, Y, Z = 1e-10, 1e-10, 1e-10
    correct_bujian, predict_bujian, gold_bujian = 1e-10, 1e-10, 1e-10
    correct_xingneng, predict_xingneng, gold_xingneng = 1e-10, 1e-10, 1e-10
    correct_jiance, predict_jiance, gold_jiance = 1e-10, 1e-10, 1e-10
    correct_zucheng, predict_zucheng, gold_zucheng = 1e-10, 1e-10, 1e-10

    f = open('./save/badcase.json', 'w', encoding='utf8')
    bujian = 0
    xingneng = 0
    jiance = 0
    zucheng = 0

    for d in tqdm(data, desc='Evaluation', total=len(data)):
        R = set([SPO(spo) for spo in extract_spoes(args, d['text'], threshold=0.0, model=model)])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)  # 抽取三元组和标注三元组匹配的个数，包括h.name,t.name,h.pos,t.pos以及relation都相同
        Y += len(R)  # 抽取三元组个数
        Z += len(T)  # 标注三元组个数


        bujian_pred, bujian_gold = [], []
        xingneng_pred, xingneng_gold = [], []
        jiance_pred, jiance_gold = [], []
        zucheng_pred, zucheng_gold = [], []
        for item in list(R):
            if item[2] == '部件故障':
                bujian_pred.append((item[0], item[1], item[-2], item[-1]))
            elif item[2] == '性能故障':
                xingneng_pred.append((item[0], item[1], item[-2], item[-1]))
            elif item[2] == "检测工具":
                jiance_pred.append((item[0], item[1], item[-2], item[-1]))
            else:
                zucheng_pred.append((item[0], item[1], item[-2], item[-1]))

        for dom in list(T):
            if dom[2] == '部件故障':
                bujian += 1
                bujian_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            elif dom[2] == '性能故障':
                xingneng += 1
                xingneng_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            elif dom[2] == '检测工具':
                jiance += 1
                jiance_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            else:
                zucheng += 1
                zucheng_gold.append((dom[0], dom[1], dom[-2], dom[-1]))

        correct_bujian += len([t for t in bujian_pred if t in bujian_gold])
        predict_bujian += len(bujian_pred)
        gold_bujian += len(bujian_gold)

        correct_xingneng += len([t for t in xingneng_pred if t in xingneng_gold])
        predict_xingneng += len(xingneng_pred)
        gold_xingneng += len(xingneng_gold)

        correct_jiance += len([t for t in jiance_pred if t in jiance_gold])
        predict_jiance += len(jiance_pred)
        gold_jiance += len(jiance_gold)

        correct_zucheng += len([t for t in zucheng_pred if t in zucheng_gold])
        predict_zucheng += len(zucheng_pred)
        gold_zucheng += len(zucheng_gold)


        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')


    bujian_p = correct_bujian / predict_bujian
    bujian_r = correct_bujian / gold_bujian
    bujian_f = 2 * bujian_p * bujian_r / (bujian_p + bujian_r)

    xingneng_p = correct_xingneng / predict_xingneng
    xingneng_r = correct_xingneng / gold_xingneng
    xingneng_f = 2 * xingneng_p * xingneng_r / (xingneng_p + xingneng_r)

    jiance_p = correct_jiance / predict_jiance
    jiance_r = correct_jiance / gold_jiance
    jiance_f = 2 * jiance_p * jiance_r / (jiance_p + jiance_r)

    zucheng_p = correct_zucheng / predict_zucheng
    zucheng_r = correct_zucheng / gold_zucheng
    zucheng_f = 2 * zucheng_p * zucheng_r / (zucheng_p + zucheng_r)
    model.train()
    logging.info(f'BJf1:  {bujian_f} XNf1: {xingneng_f} JCf1: {jiance_f} ZCf1: {zucheng_f}')
    micro_f1 = (bujian_f * bujian + xingneng_f * xingneng + jiance_f * jiance + zucheng_f * zucheng) / (bujian + xingneng + jiance + zucheng)
    f.close()
    return micro_f1

def compute_gp_kl_loss(logits_a, logits_b):
    b, h = logits_a.shape[0] , logits_a.shape[1]
    logits_a = torch.reshape(logits_a, (b, h, -1))  # [batch_size * num_labels, seq_len * seq_len]
    logits_b = torch.reshape(logits_b, (b, h, -1))

    sig_logits_a  = torch.sigmoid(logits_a)  # [batch_size * num_labels, seq_len * seq_len]
    sig_logits_b = torch.sigmoid(logits_b)

    logits_diff = logits_a - logits_b  # [batch_size * num_labels, seq_len * seq_len]
    sig_logits_diff = sig_logits_a - sig_logits_b
    kl_loss = torch.sum(torch.mul(logits_diff, sig_logits_diff), -1)   # [batch_size * num_labels]

    return kl_loss.mean()

def train(args, fold):
    os.makedirs(args.savemodel_path, exist_ok = True)
    device = args.device
    train_dataloader, val_dataloader, val_data = data_helper(args, tokenizer)
    args.max_steps = len(train_dataloader) * args.max_epochs
    args.warmup_steps = args.max_steps * 0.1

    model = ERENet(args).to(device)
    optimizer, scheduler = build_optimizer(args, model)

    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    if args.fgm == True:
        fgm = FGM(model)
    if args.ema == True:
        ema = EMA(model, 0.995)
        ema.register()
    total_loss, total_f1 = 0., 0.
    total_step = 0
    best_f1 = args.best_score
    lossed = []
    for epoch in range(1, min(30, args.max_epochs + 1)):
        logging.info(f'=========================Epoch{epoch}=======================')
        for batch in tqdm(train_dataloader):
            total_step += 1
            with autocast():
                text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
                batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(
                        device), batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
                logits1, logits2, logits3 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
                if args.r_drop == True:
                    logits1_r, logits2_r, logits3_r = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)

                loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1, mask_zero=True)
                loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2, mask_zero=True)
                loss3 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3, mask_zero=True)

                if args.r_drop == True:
                    loss1_r = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1_r, mask_zero=True)
                    loss2_r = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2_r, mask_zero=True)
                    loss3_r = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3_r, mask_zero=True)

                    loss_r_drop1 = compute_gp_kl_loss(logits1, logits1_r)
                    loss_r_drop2 = compute_gp_kl_loss(logits2, logits2_r)
                    loss_r_drop3 = compute_gp_kl_loss(logits3, logits3_r)

                    loss = ((1.5 * loss1 + loss2 + loss3) / 3 + (1.5 * loss1_r + loss2_r + loss3_r) / 3) / 2
                    loss += (loss_r_drop1 + loss_r_drop2 + loss_r_drop3) / 30
                else:
                    loss = 1.5 * loss1 + loss2 + loss3
                    loss /= 3

            scaler.scale(loss).backward()

            if args.fgm == True and epoch > args.trick_epoch:
                fgm.attack()
                with autocast():
                    logits1_adv, logits2_adv, logits3_adv = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
                    loss1_adv = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1_adv,
                                                                           mask_zero=True)
                    loss2_adv = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2_adv,
                                                                           mask_zero=True)
                    loss3_adv = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3_adv,
                                                                           mask_zero=True)
                    loss_adv = (1.5 * loss1_adv + loss2_adv + loss3_adv) / 3
                scaler.scale(loss_adv).backward()
                fgm.restore()

            scaler.step(optimizer)
            optimizer.zero_grad()
            scheduler.step()
            scaler.update()
            if args.ema == True and epoch > args.trick_epoch:
                ema.update()

            total_loss += loss.item()
            lossed.append(loss.item())
        if epoch >= 5:
            if args.ema == True and epoch > args.trick_epoch:
                ema.apply_shadow()
            logging.info(f'loss = {np.mean(lossed)}')
            lossed = []
            f = evaluate(args, val_data, model)
            if f > best_f1:
                best_f1 = f
                torch.save(model.state_dict(), f'./save/fold{fold}_best.bin')
            logging.info(f'The f1 score is: {f}, The Best f1 score is {best_f1}')
            if args.ema == True and epoch > args.trick_epoch:
                ema.restore()

def val(args):
    args.train_json = './data/1train.json'
    args.valid_json = './data/new_train_baseline_with_test.json'
    train_dataloader, val_dataloader, val_data = data_helper(args, tokenizer)
    model = ERENet(args).to('cuda')
    model.load_state_dict(torch.load(args.ckpt_file, map_location='cpu'))
    f = evaluate(args, val_data, model)
    print(f'The f1 score is {f}')

if __name__ == '__main__':
    args = parse_args()
    setup_seed(args)
    setup_device(args)
    setup_logging()
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir, do_lower_case=True)
    logging.info('======================fold1===================')
    args.train_json = './data/1train.json'
    args.valid_json = './data/1valid.json'
    train(args, 1)
    logging.info('======================fold2===================')
    args.train_json = './data/2train.json'
    args.valid_json = './data/2valid.json'
    train(args, 2)
    logging.info('======================fold3===================')
    args.train_json = './data/3train.json'
    args.valid_json = './data/3valid.json'
    train(args, 3)
    logging.info('======================fold4===================')
    args.train_json = './data/4train.json'
    args.valid_json = './data/4valid.json'
    train(args, 4)
    logging.info('======================fold5===================')
    args.train_json = './data/5train.json'
    args.valid_json = './data/5valid.json'
    train(args, 5)
#     val(args)
