import torch
import torch.nn as nn

from audio_datasets import *
from audio_models import *
from audio_feat_processing import *
from audio_train import *


with open("./audio_train.yaml", "r") as f:
    configs = yaml.safe_load(f)
    
    
def main(configs):
    s = torchaudio.datasets.IEMOCAP("./data")
    train_set, val_set, test_set = get_dataset(configs, s)
    train_loader, val_loader, test_loader = get_dataloader(configs, s)
    criterion = nn.CrossEntropyLoss()
    model = get_model(configs)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["training"]["lr"])
    best_acc = 0
    mel_extractor = Feature_extractor(configs, train_loader)
    for e in range(configs["training"]["epochs"]):
        train_one_epoch(model, train_loader, optimizer, mel_extractor, criterion, configs=configs)
        best_acc = val_one_epoch(model, val_loader, mel_extractor, save_dir=configs["training"]["save_dir"], best_acc=best_acc, configs=configs)
    test(model, test_loader, mel_extractor, save_dir=configs["training"]["save_dir"], configs=configs)

main(configs)