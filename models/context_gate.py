import torch, torch.nn as nn

class ContextGate(nn.Module):
    
    def __init__(self, ctx_dim = 128, embed_dim = 512):
        super().__init__()

        self.fc = nn.Linear(ctx_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, text_embed: torch.Tensor, ctx_embed: torch.Tensor):

        x = self.fc(ctx_embed)
        gate = self.sigmoid(x)
        
        return text_embed + gate * x