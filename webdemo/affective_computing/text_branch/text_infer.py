import os, sys
os.chdir(sys.path[0])
sys.path.append("..")
import os.path as osp
import argparse
import time
import torch
from transformers import BertTokenizerFast
from text_models import BertForSequenceClassification

def infer_sentence(args, text):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(osp.join(args.checkpoints, "best_model"))
    model = BertForSequenceClassification.from_pretrained(
        osp.join(args.checkpoints, "best_model"))
    model=model.to(device)
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
    parser.add_argument("--checkpoints", type=str, default="./infer_model")
    parser.add_argument("--gpu", type=str2bool, default=True)
    args = parser.parse_args()
    text = 'Uhuh, uhuh.  He won big and he-and he realized that the only thing that would make it better was me as his bride.'
    infer_sentence(args, text)

    print(f"done cost:{time.time()-st}")