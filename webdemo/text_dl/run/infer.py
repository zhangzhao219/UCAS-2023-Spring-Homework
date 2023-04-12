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
import json
import torch
import numpy as np
from tqdm import tqdm
from dataset.mydataset import IMDB_climate_waimai_Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
# from utils.arguments import get_args
from models.bert import BertForSequenceClassification
from utils.train_utils import compute_metrics

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

def validate(epoch, model, device, loader):
    """
    Function to be called for test with the parameters passed from main function

    """
    model.eval()
    time1=time.time()
    eval_loss = []
    pred_all = []
    label_all = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader)):
            input, mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
            
            outputs = model.forward(
                input_ids=input,
                token_type_ids=None,
                attention_mask=mask,
                labels=labels,  # 传入计算loss
            )

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            eval_loss.append(loss.item())

            logits = outputs['logits'] if isinstance(outputs, dict) else outputs[1]
            # metrics = compute_metrics(logits, labels)
            predict = torch.argmax(logits, dim=1)
            pred_all.append(predict)
            label_all.append(labels)


    pred_all = torch.concat(pred_all, dim=0).detach().cpu().numpy()
    label_all = torch.concat(label_all, dim=0).detach().cpu().numpy()
    avg_loss = np.mean(eval_loss)
    metrics = compute_metrics(pred_all, label_all)
    # {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    print(f'epoch {epoch} avg eval loss: {round(avg_loss, 6)} | acc: {metrics["accuracy"]} | p: {metrics["precision"]} | r: {metrics["recall"]} | f1: {metrics["f1"]} | ')
    return avg_loss, metrics


def main(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    # load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(osp.join(args.checkpoints, "best_model"))
    # load datasets
        # load dataset
    if args.dataset_name == 'IMDB' or args.dataset_name == 'waimai':
        args.data_dir = "../../data"
        data_path = osp.join(args.data_dir, args.dataset_name)
        test_dataset = IMDB_climate_waimai_Dataset(osp.join(data_path, "Test.csv"), tokenizer, args)
        num_labels = 2
    elif args.dataset_name in ['climate', 'IEMOCAP']:
        data_path = osp.join(args.data_dir, args.dataset_name)
        test_dataset = IMDB_climate_waimai_Dataset(osp.join(data_path, "Test.csv"), tokenizer, args)
        num_labels = 4
    else:
        assert False, f'{args.dataset_name} not exist'
    val_params = {
        "batch_size": args.val_batch_size,
        "shuffle": False,
        # "num_workers": 0,
    }
    test_loader = DataLoader(test_dataset, **val_params) 

    # load model
    model = BertForSequenceClassification.from_pretrained(
        osp.join(args.checkpoints, "best_model"))
    model=model.to(device)

    test_loss, test_metrics = validate('test', model, device, test_loader)

    with open(osp.join(args.checkpoints, 'test_metrics.json'), "w", encoding='utf-8') as f: ## 设置'utf-8'编码
        f.write(json.dumps(test_metrics, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    st=time.time()
    parser = argparse.ArgumentParser()
    # 测试模型的相关参数
    # 推理阶段checkpoints就是文件输出地址
    parser.add_argument("--checkpoints", type=str, default="../output_dir/IEMOCAP/bert-tiny/finetune/run/lr_0.0001_ep_5_bs_16_wp_0.1")
    parser.add_argument("--val_batch_size", type=int, default=128)  # 每个GPU的batch_size数
    parser.add_argument("--data_dir", type=str, default="../../data/dl_data")
    parser.add_argument("--dataset_name", type=str, default='IEMOCAP', choices=['IMDB', 'climate', 'waimai', 'IEMOCAP'])
    parser.add_argument("--max_seq_length", type=int, default=512)  #
    parser.add_argument("--max_debug_samples", type=int, default=0)
    parser.add_argument("--gpu", type=str2bool, default=True)  # 是否使用gpu
    args = parser.parse_args()
    main(args)

    print(f"done cost:{time.time()-st}")