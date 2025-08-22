import torch
import torch.nn as nn
import torch.nn.functional as F
from models.predective_base import PredectiveEncoderBase
from dpc.model_3d import DPC_RNN
from models.info_nce import DPCinfoNCE

class DPCEncoder(PredectiveEncoderBase):

    def __init__(self, sample_size: int = 112):    # Changes were made here
               
        super().__init__()

        self.sample_size = sample_size
        # self.interpolation = interpolation                              # New lines of code added 13 to 17
        # self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        # self.std  = torch.tensor([0.225,0.225,0.225]).view(1, 3, 1, 1)  


        self.dpc = DPC_RNN(
            sample_size=sample_size,       # Changed from 224 -> 112
            num_seq=6,                     # Changed from 8 -> 6
            seq_len=5,
            pred_step=3,
            network='resnet34',
        )

        state = torch.load(
            'external/DPC/model_zoo/k400_224_r34_dpc-rnn_runningStats.pth',               
            map_location="cpu",
            weights_only=False 
        )
        state_dict = state.get("state_dict", state)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.dpc.load_state_dict(state_dict, strict=False)
        
        self.backbone, self._out_dim = self.dpc.backbone, self.dpc.param['feature_size']
        self.last_duration = self.dpc.last_duration

    @property
    def out_dim(self):
        return self._out_dim
        
    def forward(self, x, *, return_pred_loss: bool = False):
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(B * T, C, H, W)  # contiguous() added
        # x = F.interpolate(
        #     x,
        #     size = (self.sample_size, self.sample_size),
        #     mode = self.interpolation,
        #     align_corners = False                                       # New lines of code 47 to 59
        # )                            

        # x = (x - self.mean.to(x)) / self.std.to(x)

        _, C, H, W = x.shape
        x = x.view(B, T, C, H, W)

        assert( T >= self.dpc.seq_len * self.dpc.num_seq ), f"Needed at least {self.dpc.seq_len*self.dpc.num_seq} frames"

        # x = x[:, :, -self.dpc.seq_len * self.dpc.num_seq :]

        S = self.dpc.num_seq * self.dpc.seq_len        
        x = x[:, -S:, :, :, :]                              # 3 new lines added here
        x = x.contiguous()

        block = x.view(B, self.dpc.num_seq, C, self.dpc.seq_len, H, W)

        # ────────────────── predictive loss ──────────────────
        if return_pred_loss:

            if self.dpc.mask is None or self.dpc.mask.size(0) != B:
                self.dpc.reset_mask()

            score, mask = self.dpc(block)
            pred_loss = DPCinfoNCE(score, mask)

            ctx_feat = self.dpc.ctx_feat

            return ctx_feat, pred_loss
        
        # ────────────────── plain extractor ──────────────────
        block = block.view(B * self.dpc.num_seq, C, self.dpc.seq_len, H, W)

        feat = self.backbone(block)
        feat = nn.functional.avg_pool3d(feat, (self.last_duration, 1, 1), stride = (1, 1, 1))
        feat = feat.mean(dim=[3, 4])
        feat = feat.squeeze(-1).view(B, self.dpc.num_seq, -1).mean(dim=[1])

        return feat