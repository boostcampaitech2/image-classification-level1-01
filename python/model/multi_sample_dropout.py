from torch import nn
import torch
import numpy as np

# age 계산 시, 일반화를 위해 사용

def multi_sample_dropout(in_feature, out_feature, p=0.5, bias=True):
    return nn.Sequential(
        nn.Dropout(p),
        nn.Linear(in_feature, out_feature, bias)
    )

def multi_sample_dropout_forward(x, dropout_layer, hidden_size=2):
    return torch.mean(torch.stack([
        dropout_layer(x) for _ in range(hidden_size)], dim=0), dim=0)

class MultiSampleDropout(nn.Module):
    def __init__(self, num_classes=3, model_name=None):
        super().__init__()
        # efficientnet_b3 와 efficientnet_b4는 in_feature가 1536
        if "efficient" in model_name:
            self.your_layer = nn.Linear(1536, 1000)
        # vit_base_patch16_224는 in_feature가 768
        elif "vit" in model_name:
            self.your_layer = nn.Linear(768, 1000)
        self.multilabel_dropout_layers = multi_sample_dropout(1000, num_classes, 0.25)
    def forward(self, x):
        x = self.your_layer(x)
        return multi_sample_dropout_forward(x, self.multilabel_dropout_layers, 4)