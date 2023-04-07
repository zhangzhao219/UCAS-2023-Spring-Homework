import torch
import torchaudio
import torch.nn as nn

from datasets import *
from models import *
from feat_processing import *


with open("./infer_config.yaml", "r") as f:
    configs = yaml.safe_load(f)
    
def main(configs):
    model = get_model(configs)
    mel_extractor = Feature_extractor(configs)
    infer_waveform = torchaudio.load(configs["data"]["dir"])[0]
    with torch.no_grad():
        model.eval()
        pred = model(mel_extractor(infer_waveform))
    return pred.argmax().item()

main(configs)