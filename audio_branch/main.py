import torch
import torch.nn as nn

from datasets import *
from models import *
from feat_processing import *
from train import *


with open("./train_config.yaml", "r") as f:
    configs = yaml.safe_load(f)
    
    
def main(configs):
    train_set, val_set, test_set = get_dataset(configs)
    train_loader, val_loader, test_loader = get_dataloader(configs)
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