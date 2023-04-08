import argparse
from utils.log_utils import log_params
import os.path as osp
import os
from typing import Optional

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def get_args():
    parser = argparse.ArgumentParser()  # 参数解释器
    # ============预训练模型参数
    # bert-base-uncased bert-large-uncased  distilbert-base-uncased prajjwal1/bert-tiny
    parser.add_argument("--model_name_or_path", type=str,default="prajjwal1/bert-tiny")  
    parser.add_argument("--cache_dir", type=str,default="/home/coder/projects/pretrain_models")

    # ============训练参数
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=8)  # 每个GPU的batch_size数
    parser.add_argument("--val_batch_size", type=int, default=8)  # 每个GPU的batch_size数
    parser.add_argument("--max_seq_length", type=int, default=512)  #
    parser.add_argument("--max_debug_samples", type=int, default=0)  
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warm up ratio')

    # ============数据集
    parser.add_argument("--data_dir", type=str, default="/home/coder/projects/Multimodal-Sentiment-Analysis/data/dl_data")
    parser.add_argument("--dataset_name", type=str, default='IEMOCAP', choices=['IMDB', 'climate', 'waimai', 'IEMOCAP'])

    # ===========文件参数
    parser.add_argument("--output_dir", type=str, default="/home/coder/projects/output_dir/IEMOCAP/bert-tiny/finetune/test")

    parser.add_argument("--step_log", type=int, default=2)
    # parser.add_argument("--val_log", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1024)

    parser.add_argument("--do_valid", type=str2bool, default=True,
                    help='predict test_set in the end of each epoch')
    parser.add_argument("--do_predict", type=str2bool, default=True,
                    help='predict test_set in the end of each epoch')
    
    
    args = parser.parse_args()  # 解析参数
    args.output_dir = osp.join(args.output_dir, f'lr_{args.lr}_ep_{args.epoch}_bs_{args.batch_size}_wp_{args.warmup_ratio}')
    # if os.path.exists(args.output_dir):  # 如果文件存在，小心覆盖

    log_params(args)
    
    return args
