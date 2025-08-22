import clip, torch

class CLIPTextEncoder():
    def __init__(self, clip_name = 'external/CLIP/ViT-B-32.pt', device = 'cpu'):
        self.device = torch.device(device)
        self.model, _ = clip.load(clip_name, device=self.device, jit=False)
        self.model.eval()

    def encode(self, texts):
        tokens = clip.tokenize(texts, truncate=True).to(self.device)

        with torch.no_grad():
            embed = self.model.encode_text(tokens)
            embed = torch.nn.functional.normalize(embed, dim=1)

        return embed