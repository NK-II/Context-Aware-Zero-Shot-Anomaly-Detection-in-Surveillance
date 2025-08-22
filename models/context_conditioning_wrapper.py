import torch
import torch.nn as nn
import torchvision.models as tv

class ContextConditioner(nn.Module):

    def __init__(self, out_dim: int = 128, freeze_backbone: bool = True):
        super().__init__()

        model = tv.resnet18(weights=None)
        state_dict = torch.load(r'external\Resnet18_Places365\resnet18_places365.pth', map_location='cpu')
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)

        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False

        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.proj = nn.Linear(512, out_dim)

    
    def forward(self, x):

        feat = self.backbone(x)
        feat = feat.flatten(1)
        ctx = self.proj(feat)

        return ctx