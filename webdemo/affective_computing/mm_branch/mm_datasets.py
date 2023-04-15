import os
import sys
import yaml
import torch
import torchaudio
import os.path as osp

from torch.utils.data import Dataset,  DataLoader
from transformers import BertTokenizerFast

os.chdir(sys.path[0])
sys.path.append('..')
from text_branch.text_dataset import IMDB_climate_waimai_Dataset
from audio_branch.audio_datasets import get_dataset


with open("./train_config.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("./text_train.yaml", "r") as f:
    text_config = yaml.safe_load(f)

class AT_Dataset(Dataset):
    def __init__(self, a_ds, t_ds):
        self.a_ds = a_ds
        self.t_ds = t_ds

    def __len__(self):
        return len(self.a_ds)

    def __getitem__(self, idx):
        item = {}
        item["audio"] = self.a_ds[idx][1]
        item['input_ids'] = self.t_ds[idx]['input_ids'].clone()
        item['attention_mask'] = self.t_ds[idx]['attention_mask'].clone()
        item["label"] = self.a_ds[idx][-1]
        return item
    
def get_mm_dataloader(train_config, text_config):
    tokenizer = BertTokenizerFast.from_pretrained(text_config["model_name_or_path"], cache_dir=text_config["cache_dir"])
    
    train_text_dataset = IMDB_climate_waimai_Dataset(osp.join(train_config["data"]["text_folder"], "Train.csv"), tokenizer)
    valid_text_dataset = IMDB_climate_waimai_Dataset(osp.join(train_config["data"]["text_folder"], "Val.csv"), tokenizer)
    test_text_dataset = IMDB_climate_waimai_Dataset(osp.join(train_config["data"]["text_folder"], "Test.csv"), tokenizer)

    s = torchaudio.datasets.IEMOCAP(train_config["data"]["audio_folder"])
    train_audio_dataset, valid_audio_dataset, test_audio_dataset = get_dataset(train_config, s)

    train_mm_dataset = AT_Dataset(train_audio_dataset, train_text_dataset)
    val_mm_dataset = AT_Dataset(valid_audio_dataset, valid_text_dataset)
    test_mm_dataset = AT_Dataset(test_audio_dataset, test_text_dataset)
    
    bs, nw = train_config["training"]["batch_size"], train_config["training"]["num_workers"]
    mm_train_loader = DataLoader(dataset=train_mm_dataset, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    mm_val_loader = DataLoader(dataset=val_mm_dataset, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False)
    mm_test_loader = DataLoader(dataset=test_mm_dataset, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False)
    
    return mm_train_loader, mm_val_loader, mm_test_loader

get_mm_dataloader(train_config, text_config)
