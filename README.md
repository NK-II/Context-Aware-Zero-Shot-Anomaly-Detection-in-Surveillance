# Context-Aware Zero-Shot Anomaly Detection in Surveillance Using Contrastive and Predictive Spatiotemporal Modeling

This repository contains the research implementation of our thesis on **Zero-Shot Anomaly Detection** in surveillance video.  
Our framework integrates **transformer-based spatiotemporal encoding (TimeSformer)**, **predictive coding via DPC-RNN**, and **semantic alignment through CLIP**.  
We further employ **context conditioning with FiLM modulation** for scene-aware generalization.  

📄 The full manuscript is included in `docs/paper.pdf`.  

---

## Key Contributions
- **Transformer-based encoder**: TimeSformer captures long-range spatiotemporal features from video sequences.  
- **Predictive coding**: DPC-RNN forecasts future representations to model temporal consistency.  
- **Semantic alignment**: CLIP-based contrastive learning bridges vision and language for zero-shot generalization.  
- **Context conditioning**: FiLM modulation dynamically adapts representations to scene attributes.  
- **Training objectives**: InfoNCE and CPC losses drive robust temporal and semantic learning.  
- **Evaluation**: Validated on the UCF-Crime dataset for anomaly detection.  

---

## Getting Started

### Installation
```bash
git clone https://github.com/NK-II/Context-Aware-Zero-Shot-Anomaly-Detection-in-Surveillance
.git
cd Context-Aware-Zero-Shot-Anomaly-Detection-in-Surveillance
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
pip install -r requirements.txt

---

### Dataset
We use the [UCF-Crime dataset](https://www.kaggle.com/) (Kaggle).  
- Place raw videos under `videos/` (ignored in `.gitignore`).  
- Data splits are provided under `data/` (train/val/test lists).  
See `DATASET.md` for setup instructions.  

---

## Repository Structure
project/
├── src/                # Core implementation
├── data/               # Split files (kept in repo)
├── videos/             # Raw dataset (not pushed, user downloads separately)
├── configs/            # Configurations
├── docs/
│   └── paper.pdf       # Full thesis / preprint
├── requirements.txt
├── README.md
└── .gitignore

```
---
## Citation
If you use this work, please cite our paper and repository:

**Paper** (preferred):
```bibtex
@article{anomaly2025,
  title   = {Context-Aware Zero-Shot Anomaly Detection in Surveillance Using Contrastive and Predictive Spatiotemporal Modeling},
  author  = {Khan, Md. Rashid Shahriar and Hasan, Md. Abrar and Justice, Mohammod Tareq Aziz},
  journal = {arXiv preprint arXiv:2508.xxxxx}, 
  year    = {2025}
}

```
**Code** (if you reference the implementation directly):
```
@misc{anomaly2025code,
  title        = {Context-Aware Zero-Shot Anomaly Detection in Surveillance Using Contrastive and Predictive Spatiotemporal Modeling},
  author       = {Khan, Md. Rashid Shahriar and Hasan, Md. Abrar and Justice, Mohammod Tareq Aziz},
  year         = {2025},
  howpublished = {\url{https://github.com/NK-II/Context-Aware-Zero-Shot-Anomaly-Detection-in-Surveillance}},
  note         = {GitHub repository}
}
```

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.






