import torch, torch.nn.functional as F

def infoNCE(video_emb, text_emb, temperature = 0.07):
    logits = (video_emb @ text_emb.t()) / temperature
    targets = torch.arange(video_emb.size(0), device=video_emb.device)
    loss_v2t = F.cross_entropy(logits, targets)
    loss_tv2 = F.cross_entropy(logits.t(), targets)

    return (loss_v2t + loss_tv2) / 2

def DPCinfoNCE(score: torch.Tensor, mask: torch.Tensor, tau: float=0.07):

    B, Pp, H2, B2, Ps, H22 = score.shape
    assert B == B2 and H2 == H22, f"Shape mismatch: B={B}, B2={B2}, H2={H2}, H22={H22}, score.shape={score.shape}"

    N_anchor = B * Pp * H2

    # ─── flatten with reshape (not view) ───────────────────────────
    logits = score.reshape(N_anchor, -1) / tau      # [N_anchor, N_cand]
    m      = mask.reshape(N_anchor, -1)             # must match logits

    # ─── compute per-anchor logZ ─────────────────────────────────
    logZ = torch.logsumexp(logits, dim=1)           # [N_anchor]

    # ─── average over all +1 entries per row ────────────────────
    pos_mask  = (m == 1).float()                    # [N_anchor, N_cand]
    pos_count = pos_mask.sum(dim=1).clamp(min=1.0)  # avoid div0
    pos_sum   = (logits * pos_mask).sum(dim=1)      # [N_anchor]
    pos_mean  = pos_sum / pos_count                 # [N_anchor]

    # ─── anchor-wise loss & mean ─────────────────────────────────
    losses = -(pos_mean - logZ)                     # [N_anchor]
    return losses.mean()