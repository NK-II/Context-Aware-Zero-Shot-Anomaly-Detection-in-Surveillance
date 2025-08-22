import torch
from models.dpc_wrapper import DPCEncoder
from models.timesformer_wrapper import TimeSformerFeatureExtractor
from models.fusion_concat import ConcateFusion
from models.mlp_projection import ProjectionMLP
from models.context_conditioning_wrapper import ContextConditioner

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timesformer = TimeSformerFeatureExtractor().to(device).eval()
    dpc = DPCEncoder().to(device).eval()
    projection = ProjectionMLP(in_dim = 768 + 256, out_dim = 512).to(device)
    context = ContextConditioner(out_dim = 128, freeze_backbone = True).to(device).eval()

    dummy_input = torch.randn(2, 3, 20, 112, 112, device=device) #[B,3,T,H,W]
    dummy_input1 = torch.randn(2, 3, 8, 224, 224, device=device) #[B,3,T,H,W]
    dummy_input2 = torch.randn(2, 3, 224, 224, device=device) #[B,3,H,W]

    with torch.no_grad():
        output1 = timesformer(dummy_input1)
        output = dpc(dummy_input)
        output2 = context(dummy_input2)

    fusion = ConcateFusion().to(device)
    fused = fusion(output1, output)
    proj = projection(fused)

    print(output.shape)
    print(output1.shape)
    print(fused.shape)
    print(proj.shape)
    print(output2.shape)

if __name__ == '__main__':
    main()