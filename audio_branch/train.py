import torch
from feat_processing import filt_aug

def train_one_epoch(model, train_loader, optimizer, mel_extractor, criterion, configs):
    model.train()
    if configs["training"]["gpu"]:
        model, mel_extractor = model.cuda(), mel_extractor.cuda()
    for idx, (_, audio, _, labels) in enumerate(train_loader):
        if configs["training"]["gpu"]:
            audio, labels = audio.cuda(), labels.cuda()
        features = mel_extractor(audio)
        if configs["training"]["filter_aug"]:
            features = filt_aug(features)
        pred = model(features)
        
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

def val_one_epoch(model, val_loader, mel_extractor, save_dir, best_acc, configs):
    with torch.no_grad():
        model.eval()
        correct_num, tot_num = 0, 0 
        if configs["training"]["gpu"]:
            model = model.cuda()
        for idx, (_, audio, _, labels) in enumerate(val_loader):
            if configs["training"]["gpu"]:
                audio, labels = audio.cuda(), labels.cuda()
            features = mel_extractor(audio)
            pred = model(features)
            _, pred_label = torch.max(pred, 1)
            correct_num += torch.sum(pred_label == labels)
            tot_num += audio.size(0)
        acc = (correct_num / tot_num).cpu().item()
        print(acc, best_acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model, save_dir)
        return best_acc

def test(model, test_loader, mel_extractor, save_dir, configs):
    correct_num, tot_num = 0, 0 
    model.load_state_dict(save_dir)
    if configs["training"]["gpu"]:
        model, mel_extractor = model.cuda(), mel_extractor.cuda()
    for idx, (_, audio, _, labels) in enumerate(test_loader):
        if configs["training"]["gpu"]:
            audio, labels = audio.cuda(), labels.cuda()
        features = mel_extractor(audio)
        pred = model(features)
        _, pred_label = torch.max(pred, 1)
        correct_num += torch.sum(pred_label == labels)
        tot_num += audio.size(0)
    acc = correct_num / tot_num
    return acc
        
        