import os, random, json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
from PIL import Image
import torchvision.transforms as T

import decord
from decord import VideoReader, cpu, gpu
decord.bridge.set_bridge("native")

def ts_to_seconds(ts: str) -> float:

    parts = ts.split(':')
    parts = [float(p) for p in parts]

    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    
    elif len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    
    else:
        raise ValueError(f"Unrecognised timestamp format {ts}")

class VideoTextDataset(Dataset):

    def __init__(self,
                manifest_path: str,
                video_root: str,
                clip_len_tsf: int = 8,      # TimeSformer 
                clip_len_dpc: int = 30,     # DPC 40 -> 30
                size_tsf: int = 224,
                size_dpc: int = 112,
                use_gpu_decode: bool = False,
                gpu_id: int = 0,
                use_dpc: bool = True
                ):
        
        self.use_dpc = use_dpc
        self.video_root = video_root
        self.clip_len_tsf = clip_len_tsf
        self.clip_len_dpc = clip_len_dpc
        self.use_gpu_decode = use_gpu_decode
        self.gpu_id = gpu_id

        self.resize_tsf = T.Resize((size_tsf, size_tsf), antialias=True)
        self.resize_dpc = T.Resize((size_dpc, size_dpc), antialias=True)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.45, 0.45, 0.45],                         # Normalization added here
            std=[0.225, 0.225, 0.225]
        )

        self.samples: List[Dict] = []
        with Path(manifest_path).open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                left, caption = line.split('##', 1)

                if len(left) == 4:
                   label, vid, t0_str, t1_str = left.strip().split()
                                                                                # changes were made here to add labels
                else:
                    label = 'normal'
                    vid, t0_str, t1_str = left.strip().split()
                
                self.samples.append(
                    {
                        "label": label.lower(),
                        "video": vid,
                        "t0": ts_to_seconds(t0_str),
                        "t1": ts_to_seconds(t1_str),
                        "caption": caption.strip(),
                    }
                )
        
        self._ctx = None


    def __getstate__(self):
        
        state = self.__dict__.copy()
        state['_ctx'] = None
        return state
    

    def __setstate__(self, state):

        self.__dict__.update(state)


    @property
    def _ctx_lazy(self):
        if self._ctx is None:
            self._ctx = gpu(self.gpu_id) if self.use_gpu_decode else cpu(0)
        return self._ctx


    def _sample_indices_random_bin(self, start_f: int, end_f: int, clip_len: int) -> List[int]:

        if clip_len <= 1 or start_f == end_f:
            return [start_f]
        
        if end_f - start_f < clip_len:
            return list(range(start_f, end_f)) + [end_f] * (clip_len - (end_f - start_f))

        bins = np.linspace(start_f, end_f, num=clip_len + 1, dtype=int)

        indices = []
        for i in range(clip_len):
            low = bins[i]
            high = min(bins[i + 1] - 1, end_f) # if error -> out of bound; change to (end_f -1)
            idx = np.random.randint(low, high + 1)
            indices.append(idx)

        return indices

        # indices= [np.random.randint(bins[i], bins[i + 1] + (1 if bins[i + 1] < end_f else 0)) 
        #         for i in range(clip_len)]
        

    
    def _sample_indices_tsf(self, start_f: int, end_f: int) -> List[int]:
        
        return self._sample_indices_random_bin(start_f, end_f, self.clip_len_tsf)


    def _sample_indices_dpc(self, start_f: int, end_f: int) -> List[int]:

        return self._sample_indices_random_bin(start_f, end_f, self.clip_len_dpc)
    
    
    def _decode_frames(self, video_path: str, indices: List[int], resize: T.Resize) -> torch.Tensor:

        vr = VideoReader(video_path, ctx=self._ctx_lazy)
        batch = vr.get_batch(indices).asnumpy()
        
        frames = []

        for frame_np in batch:
            img = Image.fromarray(frame_np)
            frame_t = self.to_tensor(resize(img))
            frame_t = self.normalize(frame_t)                # Normalized here
            frames.append(frame_t)

        return torch.stack(frames, dim=1) 

    def __len__(self):

        return len(self.samples)
    

    def __getitem__(self, idx: int):

        rec = self.samples[idx]
        video_path = os.path.join(self.video_root, f"{rec['video']}.mp4")

        vr = VideoReader(video_path, ctx=self._ctx_lazy)
        fps = vr.get_avg_fps()

        num_frames = len(vr)

        start_f = int(rec['t0'] * fps)
        end_f = min(int(rec['t1'] * fps), num_frames -1)

        #tsf
        tsf_idxs = self._sample_indices_tsf(start_f, end_f)
        clip_tsf = self._decode_frames(video_path, tsf_idxs, resize=self.resize_tsf) # [3,T,H,W]

        out = {'video_tsf' : clip_tsf}

        #context
        mid_frame_idx = tsf_idxs[len(tsf_idxs) // 2]
        mid_frame = self._decode_frames(video_path, [mid_frame_idx], resize=self.resize_tsf)[:,0] # [3,H,W]
        
        out['context_img'] = mid_frame
        #out['label'] = rec['label']  # Used only in validation

        #dpc
        if self.use_dpc:
            dpc_idxs = self._sample_indices_dpc(start_f, end_f)
            clip_dpc = self._decode_frames(video_path, dpc_idxs, resize=self.resize_dpc)

            out['video_dpc'] = clip_dpc

        out['caption'] = rec['caption']
        out["label"] = rec["label"]

        return out