import torch
import torchaudio
import yaml
import math

from torch.utils.data import Dataset, DataLoader, Subset

with open("./train_config.yaml", "r") as f:
    configs = yaml.safe_load(f)

s = torchaudio.datasets.IEMOCAP("./data")
cls2idx_mapping = {"neu":0, "hap":1, "ang":2, "sad":3}

def repeat_cut_audio(sample, configs):
    required_l = configs["data"]["audio_max_len"] * configs["data"]["sr"]
    if sample.size(1) < required_l:
        sample = sample.repeat(1, math.ceil(required_l / sample.size(1)))
    return sample[:, :required_l].flatten()

class Balanced_IEMOCAP(Dataset):
    def __init__(self, ori_iemocap, configs):
        self.unbalanced_data = ori_iemocap
        self.data = []
        cnt = 0
        for a in self.unbalanced_data:
            sample = a[0]
            new_sample = repeat_cut_audio(sample, configs)
            if a[-2] in ["neu", "hap", "ang", "sad"]:
                self.data.append((cnt, new_sample, a[-3], cls2idx_mapping[a[-2]]))
                cnt = cnt + 1
            elif a[-2] == "exc":
                self.data.append((cnt, new_sample, a[-3], cls2idx_mapping["hap"]))
                cnt = cnt + 1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_dataset(configs):
    iemocap_fullset = Balanced_IEMOCAP(s, configs)
    split_num = [round((r * len(iemocap_fullset))) for r in configs["data"]["split_ratio"]]
    train_indices, val_indices, test_indices = list(range(split_num[0])), list(range(split_num[0], split_num[0] + split_num[1])), list(range(split_num[0] + split_num[1], split_num[0] + split_num[1] + split_num[2]))
    iemocap_trainset, iemocap_valset, iemocap_testset = Subset(iemocap_fullset, train_indices), Subset(iemocap_fullset, val_indices), Subset(iemocap_fullset, test_indices)
    return iemocap_trainset, iemocap_valset, iemocap_testset

def get_dataloader(configs):
    iemocap_trainset, iemocap_valset, iemocap_testset = get_dataset(configs)
    bs, nw = configs["training"]["batch_size"], configs["training"]["num_workers"]
    train_loader = DataLoader(dataset=iemocap_trainset, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(dataset=iemocap_valset, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False)
    test_loader = DataLoader(dataset=iemocap_testset, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False)
    return train_loader, val_loader, test_loader

def get_sample_audio(configs, idx):
    st = Balanced_IEMOCAP(s, configs)
    f = st[idx][1].unsqueeze(0)
    torchaudio.save("./to_infer.wav", f, sample_rate=16000)
    
get_sample_audio(configs, 40)