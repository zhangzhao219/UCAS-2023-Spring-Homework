# -*- encoding: utf-8 -*-
"""
@Create On   : 2023/04/04 20:43:10
@Author      : CGX 
@Description : None
"""
# here put the import lib

import os, sys
import os.path as osp
os.chdir(sys.path[0])
sys.path.append('..')
import time
import numpy as np
import logging
import torch
import json
from tqdm import tqdm
from transformers import BertTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup
from utils.arguments import get_args
from models.bert import BertForSequenceClassification
from utils.train_utils import set_seed, save_model, load_mydata, compute_metrics
logger = logging.getLogger(__name__)

def predict(logits):
    '''预测函数，用于预测结果'''
    res = torch.argmax(logits, dim=1)  # 按行取每行最大的列下标
    return res

def train(epoch, model, device, loader, optimizer, scheduler):
    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    time1=time.time()
    train_loss = []
    correct = 0
    total = 0
    for _, data in enumerate(tqdm(loader)):
        input, mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
        # output中 hidden_states就是每个句子的表达  output['hidden_states']
        outputs = model.forward(
            input_ids=input,
            token_type_ids=None,
            attention_mask=mask,
            labels=labels,
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        train_loss.append(loss.item())
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[1]

        total += labels.size(0)
        correct += (predict(logits) == labels.flatten()).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.warmup_ratio != 0.0:
            scheduler.step()

        if _ % args.step_log == 0:
            time2=time.time()
            logger.info("epoch:"+str(epoch)+"-loss:"+str(round(loss.item(), 6))+";each step's time spent:"+str(float(time2-time1)/float(_+0.0001)))

    logger.info(f'epoch {epoch} avg train loss {round(np.mean(train_loss), 6)} | acc: {correct / total :.6f}')



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

            if _ % args.step_log == 0 and _!=0:
                time2=time.time()
                logger.info("epoch:"+str(epoch)+"-valid/test-loss:"+str(round(loss.item(), 6))+";each step's time spent:"+str(float(time2-time1)/float(_+0.0001)))
    
    pred_all = torch.concat(pred_all, dim=0).detach().cpu().numpy()
    label_all = torch.concat(label_all, dim=0).detach().cpu().numpy()
    avg_loss = np.mean(eval_loss)
    metrics = compute_metrics(pred_all, label_all)
    # {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    logger.info(f'epoch {epoch} avg eval loss: {round(avg_loss, 6)} | acc: {metrics["accuracy"]} | p: {metrics["precision"]} | r: {metrics["recall"]} | f1: {metrics["f1"]} | ')
    return avg_loss, metrics




def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    # load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    # load datasets
    train_loader, valid_loader, test_loader, num_labels = load_mydata(args, tokenizer)

    # load model
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir)
    
    model=model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    # warmup
    if args.warmup_ratio != 0.0:
        total_steps = len(train_loader) * args.epoch
        # total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size = 0 else (len_dataset // batch_size + 1) * epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    else:
        scheduler = None

    metrics = {}
    best_epoch = 0
    # best_loss = float('inf')
    best_f1 = 0

    for epoch in range(args.epoch):
        # 1) train for one epoch
        train(epoch, model, device, train_loader, optimizer, scheduler)

        if args.do_valid:
            # 3) evaluating test dataset
            logger.info(f"[Initiating Validation]...")
            tmp_loss, metrics = validate(epoch, model, device, valid_loader)
            # if tmp_loss < best_loss:
            #     best_loss = tmp_loss
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_epoch = epoch
                # save model for best epoch
                logger.info(f"[Saving Model]...")
                path = os.path.join(args.output_dir, 'best_model')
                save_model(model, tokenizer, path)

        logger.info(f'epoch {epoch} complete!\n')

    logger.info(f'finish training, best epoch at {best_epoch} | best f1 {best_f1}\n')
    # 测试
    if args.do_predict:
        logger.info(f"[Initiating Test]...")
        best_model = BertForSequenceClassification.from_pretrained(
            os.path.join(args.output_dir, 'best_model')
        ).to(device)
        test_loss, test_metrics = validate(f'best_epoch_{best_epoch}', best_model, device, test_loader)

    metrics['best_epoch'] = best_epoch
    metrics['best_f1'] = best_f1
    metrics['test_loss'] = test_loss
    metrics.update(test_metrics)

    with open(osp.join(args.output_dir, 'metrics.json'), "w", encoding='utf-8') as f: ## 设置'utf-8'编码
        f.write(json.dumps(metrics, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    st=time.time()
    args = get_args()
    
    main(args)

    logger.info(f"done cost:{time.time()-st}")