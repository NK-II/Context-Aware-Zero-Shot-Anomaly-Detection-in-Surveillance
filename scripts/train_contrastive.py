import yaml, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR


# ───────── internal modules ─────────
from datasets.video_text import VideoTextDataset
from models.timesformer_wrapper import TimeSformerFeatureExtractor
from models.dpc_wrapper import DPCEncoder
from models.clip_text_encoder_wrapper import CLIPTextEncoder
from models.fusion_concat import ConcateFusion
from models.mlp_projection import ProjectionMLP
from models.info_nce import infoNCE, DPCinfoNCE
from models.context_conditioning_wrapper import ContextConditioner
from models.context_gate import ContextGate
# ────────────────────────────────────

def unfreeze(module, prefixes: list[str]):
    
    for n, p in module.named_parameters():
        if any(n.startswith(pref) for pref in prefixes):
            p.requires_grad = True


def main():

    # ────────────────── inline config ──────────────────
    cfg = {
        'video_path': 'videos',
        'manifest': 'data/Train.txt',
        'batch_size': 16,
        'epochs': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    device = torch.device(cfg['device'])

    # ────────────────── mixed-precision scalar ──────────────────
    scaler = GradScaler()

    # ────────────────── dataset / loader ──────────────────
    ds = VideoTextDataset(
        manifest_path = cfg['manifest'],
        video_root = cfg['video_path']
    )
    
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # ────────────────── model ──────────────────
    tsf = TimeSformerFeatureExtractor().to(device).eval()
    
    dpc = DPCEncoder().to(device).eval()
    print("DPC_RNN has attributes:", [n for n, _ in dpc.dpc.named_children()])
    unfreeze(dpc, ['backbone.layer4', 'dpc.agg'])
    dpc.train()

    ctx = ContextConditioner(out_dim=128, freeze_backbone=True).to(device).eval()
    gate = ContextGate(ctx_dim=128, embed_dim=512).to(device) #trainable

    fusion = ConcateFusion().to(device)
    projector = ProjectionMLP(in_dim=768 + dpc.out_dim, out_dim=512 ).to(device) #trainable
    txt_encode = CLIPTextEncoder(device=device) #frozen

    # optim_proj = optim.AdamW(projector.parameters(), lr=1e-4, weight_decay=1e-3)
    # optim_all = optim.AdamW(list(projector.parameters()) + list(gate.parameters()), lr=1e-4, weight_decay=1e-3)
    optim_all = optim.AdamW([
        {'params': projector.parameters(),    'lr': 3e-4, 'weight_decay': 1e-4},
        {'params': gate.parameters(),    'lr': 3e-4, 'weight_decay': 1e-4},
        {'params': dpc.backbone.layer4.parameters(),    'lr': 3e-5, 'weight_decay': 1e-4},
        {'params': dpc.dpc.agg.parameters(),    'lr': 3e-5, 'weight_decay': 1e-4},

    ])
    # optim_all = optim.AdamW([
    #     {'params': projector.parameters(),    'lr': 1e-4},
    #     {'params': gate.parameters(),    'lr': 1e-4},
    #     {'params': dpc.backbone.layer4.parameters(),    'lr': 5e-5},
    #     {'params': dpc.dpc.agg.parameters(),    'lr': 5e-5},
    # ], weight_decay = 1e-3)

    # ────────────────── learning schedule ──────────────────
    total_steps = len(loader) * cfg['epochs']
    scheduler = CosineAnnealingLR(optim_all, T_max=total_steps)


    # ────────────────── training loop ──────────────────
    for epoch in range(cfg['epochs']):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            vid_tsf = batch['video_tsf'].to(device)  # [B,3,8,224,224]
            vid_dpc = batch['video_dpc'].to(device)  # [B,3,40,224,224]
            vid_ctx = batch['context_img'].to(device)  # [B,3,224,224]
        
            # ────────────────── forward pass ──────────────────
            with torch.no_grad():
                feat_tsf = tsf(vid_tsf)  # [B,768]  frozen
                ctx_embed = ctx(vid_ctx)  # [B,128]  frozen
            
            # with autocast(device_type="cuda", dtype=torch.float16):
            
            feat_dpc, pred_loss = dpc(vid_dpc, return_pred_loss=True)  # [B,256]  # grads into layer4 & agg

            fused = fusion(feat_tsf, feat_dpc)  # [B,1024]
            video_embed = nn.functional.normalize(projector(fused), dim=1)  # [B,512]

            # ────────────────── tokenization ──────────────────
            caps = batch['caption']
            text_embed = txt_encode.encode(caps)   # [B,512]
            # text_embed = text_embed.float()

            # ────────────────── context-conditioning ──────────────────
            text_embed = gate(text_embed, ctx_embed)  #conditioned
            text_embed = nn.functional.normalize(text_embed, dim=1)

            # ────────────────── loss & step ──────────────────
            align_loss = infoNCE(video_embed, text_embed)
            gamma = 0.9
            loss = gamma * align_loss + (1.0 - gamma) * pred_loss    
            # optim_proj.zero_grad()
            # optim_all.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.unscale_(optim_all)
            # params_to_clip = [p for p in dpc.parameters() if p.requires_grad] + list(projector.parameters()) + list(gate.parameters())
            # torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            # # optim_proj.step()
            # scaler.step(optim_all)
            # scaler.update()

            optim_all.zero_grad()
            loss.backward()
            optim_all.step()


            pbar.set_postfix(align=f"{align_loss.item():.4f}",
                            pred=f"{pred_loss.item():.4f}",
                            total=f"{loss.item():.4f}")

        # scheduler.step()

        print(f"Epoch {epoch+1} finished. loss={loss.item():.4f}")

    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(projector.state_dict(), 'checkpoints/projecor.pt')
    torch.save(gate.state_dict(), 'checkpoints/context_gate.pt')
    torch.save(dpc.state_dict(), 'checkpoints/dpc_encoder.pt')

if __name__ == '__main__':
    main()