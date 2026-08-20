#!/usr/bin/env python3
"""Regenerate t-SNE figure with Chinese labels using KG-TPCA checkpoint."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE

# Chinese font
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

from cdf_vlm.biomedclip import load_local_biomedclip
from cdf_vlm.ddsm import DDSMPairedViewDataset
from cdf_vlm.text_prompts import DEFAULT_KEYWORDS
from cdf_vlm.birads_kg import KGEnhancedPromptEncoder
from cdf_vlm.multiview_v2 import PromptAttentionGate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = Path("outputs/encoder_ablation/kg_enhanced/model.pt")
DATA_ROOT = "data/processed/ddsm"
TEST_CSV = f"{DATA_ROOT}/breast_cancer_test_cc_mlo_pair.csv"
BIOMEDCLIP_DIR = "external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OUT_DIR = Path("paper/figures/软件导刊投稿图")

print("Loading BiomedCLIP...")
clip_model, preprocess, _ = load_local_biomedclip(BIOMEDCLIP_DIR, device=DEVICE)
for p in clip_model.parameters():
    p.requires_grad = False

# Build KG-TPCA model (same as encoder ablation)
prompt_encoder = KGEnhancedPromptEncoder(
    embed_dim=512, keywords=DEFAULT_KEYWORDS,
    kg_lambda_prior=0.01, kg_lambda_pairwise=0.005,
)
prompt_gate = PromptAttentionGate(embed_dim=512)

class KGModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.prompt_gate = prompt_gate
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 1),
        )
    def encode_image(self, img):
        f = clip_model.encode_image(img)
        return torch.nn.functional.normalize(f.float(), dim=-1)
    def forward_single(self, image, description):
        feat = self.encode_image(image)
        prompt = self.prompt_encoder.encode_texts(description, device=image.device)
        feat = self.prompt_gate(feat, prompt)
        return feat  # return features, not logits

model = KGModel().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt)
model.eval()
print(f"Loaded checkpoint from {CKPT_PATH}")

# Load test data
test_dataset = DDSMPairedViewDataset(TEST_CSV, data_root=DATA_ROOT, image_transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
print(f"Test samples: {len(test_dataset)}")

# Extract CC features
features = []
labels = []
with torch.no_grad():
    for batch in test_loader:
        feat = model.forward_single(batch["cc_image"].to(DEVICE), list(batch["cc_description"]))
        features.append(feat.cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)
print(f"Features: {features.shape}, Labels: {labels.shape}")

# t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
embedded = tsne.fit_transform(features)

# Plot with Chinese labels
fig, ax = plt.subplots(figsize=(7, 5.5))

colors = {"良性": "#2196F3", "恶性": "#F44336"}
for lbl, name, color in [(0, "良性", "#2196F3"), (1, "恶性", "#F44336")]:
    mask = labels == lbl
    ax.scatter(embedded[mask, 0], embedded[mask, 1],
               c=color, label=name, alpha=0.55, s=18, edgecolors="none")

ax.set_xlabel("t-SNE 维度 1", fontsize=11)
ax.set_ylabel("t-SNE 维度 2", fontsize=11)
ax.set_title("CC视图特征空间t-SNE可视化\n（蓝色=良性，红色=恶性）", fontweight="bold", fontsize=12, pad=10)
ax.legend(loc="best", fontsize=10, markerscale=1.5)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_tsne_cc_features.png")
fig.savefig(OUT_DIR / "fig7_tsne_cc_features.pdf")
plt.close(fig)
print(f"Saved: fig7_tsne_cc_features.png/pdf to {OUT_DIR}")
