import torch
import torch.nn as nn

class ConcateFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, timesformer_feat, dpc_feat):
        return torch.cat([timesformer_feat, dpc_feat], dim=1)