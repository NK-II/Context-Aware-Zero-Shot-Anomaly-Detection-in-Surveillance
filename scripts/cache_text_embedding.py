import torch, json, argparse
from pathlib import Path

from models.clip_text_encoder_wrapper import CLIPTextEncoder
from models.context_gate import ContextGate
from models.context_conditioning_wrapper import ContextConditioner

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--captions', default='data/normal_captions.txt')
    parser.add_argument('--context_gate_ckpt', default='checkpoints/context_gate.pt')
    parser.add_argument('--out', default='data/text_embeddings.pt')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    enc = CLIPTextEncoder(device=device) 
    gate = ContextGate().to(device)

    gate.load_state_dict(torch.load(args.context_gate_ckpt, map_location=device))
    gate.eval()

    ctx_zero = torch.zeros(1, 128, device=device)

    caps = [l.strip() for l in Path(args.captions).read_text().splitlines() if l.strip]

    embeddings = []

    with torch.no_grad():
        for c in caps:
            embed = CLIPTextEncoder().encode([c])
            embed = gate(embed, ctx_zero)
            embeddings.append(torch.nn.functional.normalize(embed, dim=1))

    embeddings = torch.cat(embeddings, dim=0)

    torch.save(embeddings.cpu(), args.out)

if __name__ == '__main__':
    main()