import os
import sys
import yaml
import torch
import torchaudio
import torch.nn as nn

from affective_computing.mm_branch.mm_models import AT_Cat_Fusion

os.chdir(sys.path[0])
sys.path.append('..')

from affective_computing.audio_branch.audio_feat_processing import Feature_extractor
from transformers import BertTokenizerFast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("/app/affective_computing/mm_branch/audio_train.yaml", "r") as f:
    audio_config = yaml.safe_load(f)
with open("/app/affective_computing/mm_branch/text_train.yaml", "r") as f:
    text_config = yaml.safe_load(f)
with open("/app/affective_computing/mm_branch/train_config.yaml", "r") as f:
    train_config = yaml.safe_load(f)

def main3(raw_text):
    mm_model = AT_Cat_Fusion(text_config, audio_config)
    weight_dir = "/app/affective_computing/mm_branch/ckpt/best_model.pth" 
    state_dict = torch.load(weight_dir, map_location=torch.device('cpu'))
    mm_model.load_state_dict(state_dict)
    mm_model = mm_model.to(device)
    mel_extractor = Feature_extractor(train_config)
    tokenizer = BertTokenizerFast.from_pretrained(text_config['model_name_or_path'], cache_dir=text_config["cache_dir"])
    
    raw_audio = torchaudio.load("/app/wav/to_infer.wav")[0]
    audio_input = mel_extractor(raw_audio).to(device)
    # raw_text = "Uhuh, uhuh. He won big and he-and he realized that the only thing that would make it better was me as his bride."
    text_input = tokenizer(raw_text, padding="longest", return_tensors="pt").to(device)
    with torch.no_grad():
        mm_model.eval()
        pred = mm_model(audio_input, text_input['input_ids'], text_input['attention_mask'])
        print(pred)
    return pred.argmax().item()

# print(main3())