import torch

ckpt = torch.load('external\DPC\model_zoo\k400_224_r34_dpc-rnn_runningStats.pth', map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']

# Lists all layer names and their weight shapes
for k, v in state_dict.items():
    print(f"{k}: {tuple(v.shape)}")


print("Top-level keys:", ckpt.keys())

