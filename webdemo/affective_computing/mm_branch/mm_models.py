import os
import yaml
import sys
import torch
import torch.nn as nn


os.chdir(sys.path[0])
sys.path.append('..')

from affective_computing.text_branch.text_models import BertForSequenceClassification
from affective_computing.audio_branch.audio_models import get_model
    
class AT_Cat_Fusion(nn.Module):
    def __init__(self, text_config, audio_config, nclass=4):
        super(AT_Cat_Fusion, self).__init__()
        self.emb_dim = 128
        self.text_encoder = BertForSequenceClassification.from_pretrained(text_config['model_name_or_path'], num_labels=nclass, cache_dir=text_config["cache_dir"])
        self.audio_encoder = get_model(audio_config)
        self.text_projector = nn.Linear(self.text_encoder.classifier.in_features, self.emb_dim)
        self.audio_projector = nn.Linear(self.audio_encoder.fc1.in_features, self.emb_dim)
        self.cls_head = nn.Linear(2 * self.emb_dim, nclass)
        self.sf = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        self.load_init_params(audio_config)
        
    def load_init_params(self, audio_config):
        state_dict = torch.load(audio_config["training"]["pretrain_dir"], map_location=torch.device('cpu'))
        self.audio_encoder.load_state_dict(state_dict, strict=False)
    
    def forward(self, a, t_ids, t_mask):
        text_feat = self.text_encoder(input_ids=t_ids, token_type_ids=None, attention_mask=t_mask)["hidden_states"]
        text_embed = self.text_projector(text_feat)
        audio_feat = self.audio_encoder(a, return_feat=True)
        audio_embed = self.audio_projector(audio_feat)
        mm_embed = self.dropout(torch.cat([audio_embed, text_embed], dim=1))
        cls_probs = self.cls_head(mm_embed)
        return self.sf(cls_probs)

class AT_Attention_Fusion(nn.Module):
    def __init__(self, text_config, audio_config, nclass=4):
        super(AT_Attention_Fusion, self).__init__()
        self.audio_emb_dim = 128
        self.emb_dim = 128
        self.text_encoder = BertForSequenceClassification.from_pretrained(text_config['model_name_or_path'], num_labels=nclass, cache_dir=text_config["cache_dir"])
        self.audio_encoder = get_model(audio_config)
        self.text_projector = nn.Linear(self.text_encoder.classifier.in_features, self.emb_dim)
        self.audio_projector = nn.Linear(self.audio_emb_dim, self.emb_dim)
        self.audio_fc = nn.Linear(self.audio_encoder.fc1.in_features, self.audio_emb_dim)
        self.audio_attn = nn.MultiheadAttention(self.audio_emb_dim, num_heads=4, batch_first=True)
        self.cls_head = nn.Linear(2 * self.emb_dim, nclass)
        self.sf = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        self.load_init_params(audio_config)
        
    def load_init_params(self, audio_config):
        state_dict = torch.load(audio_config["training"]["pretrain_dir"], map_location=torch.device('cpu'))
        self.audio_encoder.load_state_dict(state_dict, strict=False)
    
    def forward(self, a, t_ids, t_mask):
        text_feat = self.text_encoder(input_ids=t_ids, token_type_ids=None, attention_mask=t_mask)["hidden_states"]
        text_embed = self.text_projector(text_feat)
        audio_feat_fr = self.dropout(self.audio_fc(self.audio_encoder(a, return_fr_feat=True)))
        audio_aggregate_feat, _ = self.audio_attn(query=text_embed.unsqueeze(1), key=audio_feat_fr, value=audio_feat_fr)
        audio_embed = self.audio_projector(audio_aggregate_feat.squeeze(1))
        mm_embed = self.dropout(torch.cat([audio_embed, text_embed], dim=1))
        cls_probs = self.cls_head(mm_embed)
        return self.sf(cls_probs)
        

# AT_Cat_Fusion(text_config, audio_config)