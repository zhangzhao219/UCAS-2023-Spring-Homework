import yaml
import torch
import torchaudio
import torch.nn as nn

from audio_models import *
from audio_feat_processing import *



    
def main():
    with open("/app/affective_computing/audio_branch/audio_infer.yaml", "r") as f:
        configs = yaml.safe_load(f)
    model = get_model(configs)
    weight_dir = "/app/affective_computing/audio_branch/ckpt/best_model.pth"
    state_dict = torch.load(weight_dir, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    mel_extractor = Feature_extractor(configs)
    infer_waveform = torchaudio.load(configs["data"]["dir"])[0]
    with torch.no_grad():
        model.eval()
        feat = mel_extractor(infer_waveform)
        pred = model(feat)
        print(pred)
    return pred.argmax().item()

# print(main(configs))