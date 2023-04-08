import numpy as np 
import random 
import torch 
import os 
import logging
import os.path as osp
from dataset.mydataset import IMDB_climate_waimai_Dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def compute_metrics(pred, labels):
    # pred = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    # pred = np.argmax(pred, axis=1)

    accuracy = round(accuracy_score(y_true=labels, y_pred=pred), 6)
    recall = round(recall_score(y_true=labels, y_pred=pred, average='macro', zero_division=0), 6)
    precision = round(precision_score(y_true=labels, y_pred=pred, average='macro', zero_division=0), 6)
    f1 = round(f1_score(y_true=labels, y_pred=pred, average='macro', zero_division=0), 6)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

logger = logging.getLogger(__name__)

def save_model(model, tokenizer, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    #如果我们有一个分布式模型，只保存封装的模型
    #它包装在PyTorch DistributedDataParallel或DataParallel中
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)
    # torch.save(model_to_save.state_dict(), osp.join(path, 'pytorch_model.bin'))
    # model_to_save.config.to_json_file(osp.join(path, 'config.json'))
    # tokenizer.save_vocabulary(path)
    tokenizer.save_pretrained(path)

def load_mydata(args, tokenizer):
    print("load datasets ...")
    # load dataset
    if args.dataset_name == 'IMDB' or args.dataset_name == 'waimai':
        data_path = osp.join(args.data_dir, args.dataset_name)
        train_dataset = IMDB_climate_waimai_Dataset(osp.join(data_path, "Train.csv"), tokenizer, args)
        valid_dataset = IMDB_climate_waimai_Dataset(osp.join(data_path, "Val.csv"), tokenizer, args)
        test_dataset = IMDB_climate_waimai_Dataset(osp.join(data_path, "Test.csv"), tokenizer, args)
        num_labels = 2
    elif args.dataset_name in ['climate', 'IEMOCAP']:
        data_path = osp.join(args.data_dir, args.dataset_name)
        train_dataset = IMDB_climate_waimai_Dataset(osp.join(data_path, "Train.csv"), tokenizer, args)
        valid_dataset = IMDB_climate_waimai_Dataset(osp.join(data_path, "Val.csv"), tokenizer, args)
        test_dataset = IMDB_climate_waimai_Dataset(osp.join(data_path, "Test.csv"), tokenizer, args)
        num_labels = 4
    else:
        assert False, f'{args.dataset_name} not exist'

    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        # "num_workers": 0,
    }
    
    val_params = {
        "batch_size": args.val_batch_size,
        "shuffle": False,
        # "num_workers": 0,
    }
    
    train_loader = DataLoader(train_dataset, **train_params)
    valid_loader = DataLoader(valid_dataset, **val_params)  
    test_loader = DataLoader(test_dataset, **val_params) 
    return train_loader, valid_loader, test_loader, num_labels


def set_seed(seed: int = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")