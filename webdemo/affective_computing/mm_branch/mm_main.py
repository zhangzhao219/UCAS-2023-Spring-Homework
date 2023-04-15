import os
import torch
import torch.nn as nn
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from mm_datasets import *
from mm_models import *
from audio_branch.audio_feat_processing import *
from mm_train import *


with open("./train_config.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("./audio_train.yaml", "r") as f:
    audio_config = yaml.safe_load(f)
with open("./text_train.yaml", "r") as f:
    text_config = yaml.safe_load(f)
    
def main(train_config, audio_config, text_config):
    train_loader, val_loader, test_loader = get_mm_dataloader(train_config, text_config)
    criterion = nn.CrossEntropyLoss()
    if train_config["training"]["model_type"] == "cat":
        mm_model = AT_Cat_Fusion(text_config, audio_config)
    elif train_config["training"]["model_type"] == "attention":
        mm_model = AT_Attention_Fusion(text_config, audio_config)
    optimizer = torch.optim.AdamW(mm_model.parameters(), lr=train_config["training"]["lr"])
    best_acc = 0
    mel_extractor = Feature_extractor(train_config)
    for e in range(train_config["training"]["epochs"]):
        train_one_epoch(mm_model, train_loader, optimizer, mel_extractor, criterion, configs=train_config)
        best_acc = val_one_epoch(mm_model, val_loader, mel_extractor, save_dir=train_config["training"]["save_dir"], best_acc=best_acc, configs=train_config)
    test(mm_model, test_loader, mel_extractor, save_dir=train_config["training"]["save_dir"], configs=train_config)

main(train_config, audio_config, text_config)