import torch.nn as nn

class PredectiveEncoderBase(nn.Module):

    @property
    def out_dim(self) -> int:
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
