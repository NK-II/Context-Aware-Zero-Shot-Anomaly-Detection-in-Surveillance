import torch.nn as nn 

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, out_dim),
            nn.LayerNorm(out_dim)
        )


    def forward(self, x):
        return self.mlp(x)