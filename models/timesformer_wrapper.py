import torch
import torch.nn as nn
from scripts.path_config import timesformer_root
from timesformer.models.vit import TimeSformer
from models.backbone_base import BackboneBase

class TimeSformerFeatureExtractor(BackboneBase):

    def __init__(self):
        super().__init__()

        self.timesformer = TimeSformer(
        img_size = 244,
        patch_size = 16,
        num_classes = 400,
        num_frames = 8,
        attention_type = 'divided_space_time',
        pretrained_model = str(timesformer_root / 'TimeSformer_divST_8x32_224_K400.pyth'),            
        )

        self._out_dim = self.timesformer.model.embed_dim
        self.timesformer.model.head = nn.Identity()

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, x):
        return self.timesformer(x)