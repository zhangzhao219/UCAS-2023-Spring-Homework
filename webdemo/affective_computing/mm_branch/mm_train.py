import torch
from audio_feat_processing import filt_aug, mixup, spec_aug
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def train_one_epoch(model, train_loader, optimizer, mel_extractor, criterion, configs):
    model.train()
    if configs["training"]["gpu"]:
        model, mel_extractor = model.cuda(), mel_extractor.cuda()
    for idx, data in enumerate(train_loader):
        audio, input_ids, attn_masks, labels = data['audio'], data['input_ids'], data['attention_mask'], data['label']
        if configs["training"]["gpu"]:
            audio, input_ids, attn_masks, labels = audio.cuda(), input_ids.cuda(), attn_masks.cuda(), labels.cuda()
        features = mel_extractor(audio)
        if configs["training"]["filter_aug"]:
            features = filt_aug(features)
        if configs["training"]["spec_aug"]:
            features = spec_aug(features) 
        pred = model(features, input_ids, attn_masks)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

def val_one_epoch(model, val_loader, mel_extractor, save_dir, best_acc, configs):
    with torch.no_grad():
        model.eval()
        tot_pred_labels, tot_gt = None, None
        if configs["training"]["gpu"]:
            model = model.cuda()
        for idx, data in enumerate(val_loader):
            audio, input_ids, attn_masks, labels = data['audio'], data['input_ids'], data['attention_mask'], data['label']
            if configs["training"]["gpu"]:
                audio, input_ids, attn_masks, labels = audio.cuda(), input_ids.cuda(), attn_masks.cuda(), labels.cuda()
            features = mel_extractor(audio)
            pred = model(features, input_ids, attn_masks)
            _, pred_label = torch.max(pred, 1)
            if tot_pred_labels is None:
                tot_pred_labels = pred_label
                tot_gt = labels
            else:
                tot_pred_labels = torch.cat([tot_pred_labels, pred_label])
                tot_gt = torch.cat([tot_gt, labels])
                
        tot_pred_labels_np = tot_pred_labels.flatten().detach().cpu().numpy()
        tot_gt_np = tot_gt.flatten().detach().cpu().numpy()
        
        ua = accuracy_score(tot_gt_np, tot_pred_labels_np)
        wa = balanced_accuracy_score(tot_gt_np, tot_pred_labels_np)
        if ua + wa > best_acc:
            best_acc = ua + wa
            torch.save(model.state_dict(), save_dir)
        print("UA = {:.3f}, WA = {:.3f}, UA + WA = {:.3f}. Best UA + WA = {:.3f}.".format(ua, wa, ua + wa, best_acc))
        return best_acc

def test(model, test_loader, mel_extractor, save_dir, configs):
    with torch.no_grad():
        model.eval()
        tot_pred_labels, tot_gt = None, None
        state_dict = torch.load(save_dir, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        if configs["training"]["gpu"]:
            model, mel_extractor = model.cuda(), mel_extractor.cuda()
        for idx, data in enumerate(test_loader):
            audio, input_ids, attn_masks, labels = data['audio'], data['input_ids'], data['attention_mask'], data['label']
            if configs["training"]["gpu"]:
                audio, input_ids, attn_masks, labels = audio.cuda(), input_ids.cuda(), attn_masks.cuda(), labels.cuda()
            features = mel_extractor(audio)
            pred = model(features, input_ids, attn_masks)
            _, pred_label = torch.max(pred, 1)
            if tot_pred_labels is None:
                tot_pred_labels = pred_label
                tot_gt = labels
            else:
                tot_pred_labels = torch.cat([tot_pred_labels, pred_label])
                tot_gt = torch.cat([tot_gt, labels])
                
        tot_pred_labels_np = tot_pred_labels.flatten().detach().cpu().numpy()
        tot_gt_np = tot_gt.flatten().detach().cpu().numpy()
        
        ua = accuracy_score(tot_gt_np, tot_pred_labels_np)
        wa = balanced_accuracy_score(tot_gt_np, tot_pred_labels_np)
        
        print("UA = {:.3f}, WA = {:.3f}, UA + WA = {:.3f}.".format(ua, wa, ua + wa))
        
        