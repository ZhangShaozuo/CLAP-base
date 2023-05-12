import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers import Wav2Vec2Model, BertModel
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
### Todo:
# 1. Use CLAP-MOCO

def cross_entropy(preds, targets, reduction = 'none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
@dataclass
class CLAPOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    l_last_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    a_last_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ProjectionHead(nn.Module):
    def __init__(self, config, pooled):
        super(ProjectionHead, self).__init__()
        self.pooled = pooled
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(config['hidden_size'], config['projection_size'])
        self.gelu = nn.GELU()
        self.fc = nn.Linear(config['projection_size'], config['projection_size'])
        self.dropout = nn.Dropout(config['train']['dropout'])
        self.layer_norm = nn.LayerNorm(config['projection_size'])
        
    def forward(self, x):
        if self.pooled:
            x = self.pool(x).squeeze(2)
        prj_x = self.projection(x)
        x = self.gelu(prj_x)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + prj_x
        x = self.layer_norm(x)
        return x


class CLAP(nn.Module):
    'Language Audio Pair for style embedding'
    def __init__(self, config) -> None:
        super(CLAP, self).__init__()
        self.config = config
        self.l_model = BertModel.from_pretrained(config['language_model']['ckpt'])
        self.a_model = Wav2Vec2Model.from_pretrained(config['audio_model']['ckpt'])
        self.a_model.freeze_feature_encoder()
        self.l_projector = ProjectionHead(config, pooled = False)
        self.a_projector = ProjectionHead(config, pooled = True)
        
    def forward(self, input_ids, l_attn_mask, input_values, a_attn_mask):
        l_embeds = self.l_model(input_ids, attention_mask = l_attn_mask)
        a_embeds = self.a_model(input_values, attention_mask = a_attn_mask)
        
        l_embeds = self.l_projector(l_embeds.pooler_output)
        a_embeds = self.a_projector(a_embeds.last_hidden_state.transpose(1,2))
        logits = l_embeds @ a_embeds.T
        # check the value range of l_embeds and a_embeds
        a_similarities = a_embeds @ a_embeds.T
        l_similarities = l_embeds @ l_embeds.T
        targets = F.softmax(
            (a_similarities + l_similarities)/2, dim=-1
        )
        loss_a = cross_entropy(logits, targets, reduction = 'none')
        loss_t = cross_entropy(logits.T, targets.T, reduction = 'none')
        loss = (loss_a + loss_t)/2

        return CLAPOutput(
            loss = loss.mean(),
            logits = logits,
            l_last_hidden_states = l_embeds,
            a_last_hidden_states = a_embeds
        )