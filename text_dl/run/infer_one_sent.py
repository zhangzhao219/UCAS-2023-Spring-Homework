# -*- encoding: utf-8 -*-
"""
@Create On   : 2023/04/06 20:49:08
@Author      : CGX 
@Description : None
"""
# here put the import lib
import os, sys
os.chdir(sys.path[0])
sys.path.append("..")
import os.path as osp
import argparse
import time
import torch
from transformers import BertTokenizerFast
from models.bert import BertForSequenceClassification

def infer_sentence(args, text):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    # load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(osp.join(args.checkpoints, "best_model"))
    # load model
    model = BertForSequenceClassification.from_pretrained(
        osp.join(args.checkpoints, "best_model"))
    model=model.to(device)
    # text 可以是str，list[str]
    data = tokenizer(text, padding="longest", return_tensors="pt").to(device)
    output = model(**data)
    text_hidden = output['hidden_states']



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

if __name__ == '__main__':
    st=time.time()
    parser = argparse.ArgumentParser()
    # 测试模型的相关参数
    parser.add_argument("--checkpoints", type=str, default="/home/coder/projects/output_dir/IEMOCAP/bert-tiny/finetune/run/lr_0.0001_ep_5_bs_16_wp_0.1")
    parser.add_argument("--gpu", type=str2bool, default=True)
    args = parser.parse_args()
    text = 'hey zhangzhao'
    infer_sentence(args, text)

    print(f"done cost:{time.time()-st}")