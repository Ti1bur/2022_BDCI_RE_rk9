import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Final Code for 2022BDCI RE")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_json', type=str, default='./data/train_data.json')
    parser.add_argument('--valid_json', type=str, default='./data/eval_data.json')
    parser.add_argument('--test_output_csv', type=str, default='./data/result.json')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=4, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=64, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=64, type=int, help="use for testing duration per worker")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--ckpt_file', type=str, default='./save/fold5/nezha-wwm/fold1_epoch18_f1_0.7121118825454591.bin')
    parser.add_argument('--best_score', default=0.65, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=100, help='How many epochs')
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=4e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")


    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='/WCBD_7_tmp/pretrain_model/nezha-large-wwm')
    parser.add_argument('--bert_cache', type=str, default='./save/cache')
    parser.add_argument('--bert_seq_length', type=int, default=320)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Other =============================
    parser.add_argument('--hidden_size', type=int, default=1024, help="bert_hidden_size")
    parser.add_argument('--inner_dim', type=int, default=64, help="RawGlobalPointer_inner_dim")
    parser.add_argument('--savemodel_path', type=str, default='./save')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--fgm', type=bool, default=True)
    parser.add_argument('--trick_epoch', type=int, default=0)
    parser.add_argument('--r_drop', type=bool, default=False)
    parser.add_argument('--fake_label', type=bool, default=False)

    return parser.parse_args()