#!/usr/bin/env python3
"""BiomedCLIP zero-shot classification on CBIS-DDSM test set.
Uses BiomedCLIP's native text encoder with medical prompts - no training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cdf_vlm.biomedclip import load_local_biomedclip
from cdf_vlm.config import save_json
from cdf_vlm.ddsm import DDSMPairedViewDataset
from cdf_vlm.io import ensure_dir
from cdf_vlm.metrics import binary_metrics

DATA_ROOT = "data/processed/ddsm"
TEST_PAIR_CSV = f"{DATA_ROOT}/breast_cancer_test_cc_mlo_pair.csv"
BIOMEDCLIP_DIR = "external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OUT_DIR = Path("outputs/sota_baselines/biomedclip_zeroshot")

# Medical-domain prompts for mammogram classification
TEMPLATES = [
    "this is a mammogram showing a {} lesion",
    "this is a mammogram with {} findings",
    "a {} breast lesion on mammography",
]
CLASSES = ["benign", "malignant"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_dir = ensure_dir(OUT_DIR)
    print(f"Output: {out_dir}")

    # Load BiomedCLIP with tokenizer (need text branch!)
    print("Loading BiomedCLIP...")
    clip_model, preprocess, tokenizer = load_local_biomedclip(BIOMEDCLIP_DIR, device=device)
    clip_model = clip_model.to(device).eval()

    # Build zero-shot prompts
    prompts = []
    for tmpl in TEMPLATES:
        for cls_name in CLASSES:
            prompts.append(tmpl.format(cls_name))
    print(f"Zero-shot prompts ({len(prompts)}):")
    for p in prompts:
        print(f"  '{p}'")

    # Encode all text prompts once
    with torch.no_grad():
        text_tokens = tokenizer(prompts, context_length=256).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    print(f"Text features: {text_features.shape}")

    # Organize by class (benign indices: 0,2,4; malignant: 1,3,5)
    benign_feat = text_features[0::2].mean(dim=0, keepdim=True)    # average over templates
    malignant_feat = text_features[1::2].mean(dim=0, keepdim=True)
    class_text_features = torch.cat([benign_feat, malignant_feat], dim=0)
    # Row 0 = benign, Row 1 = malignant
    print(f"Class text features: {class_text_features.shape}")

    # Load test data
    test_dataset = DDSMPairedViewDataset(TEST_PAIR_CSV, data_root=DATA_ROOT, image_transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    print(f"Test samples: {len(test_dataset)}")

    # ── Single-view (CC only) zero-shot ──
    cc_probs, cc_labels = [], []
    # ── Dual-view (CC + MLO average) zero-shot ──
    dual_probs, dual_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Zero-shot eval"):
            labels = batch["label"].cpu().numpy()

            # CC view
            cc_img = batch["cc_image"].to(device)
            cc_feat = clip_model.encode_image(cc_img)
            cc_feat = torch.nn.functional.normalize(cc_feat.float(), dim=-1)
            cc_logits = cc_feat @ class_text_features.T
            cc_prob = torch.softmax(cc_logits, dim=-1)[:, 1].cpu().numpy()  # malignant prob

            # MLO view
            mlo_img = batch["mlo_image"].to(device)
            mlo_feat = clip_model.encode_image(mlo_img)
            mlo_feat = torch.nn.functional.normalize(mlo_feat.float(), dim=-1)
            mlo_logits = mlo_feat @ class_text_features.T
            mlo_prob = torch.softmax(mlo_logits, dim=-1)[:, 1].cpu().numpy()

            cc_probs.extend(cc_prob.tolist())
            dual_probs.extend(((cc_prob + mlo_prob) / 2.0).tolist())
            cc_labels.extend(labels.tolist())
            dual_labels.extend(labels.tolist())

    # Metrics
    cc_metrics = binary_metrics(cc_labels, cc_probs)
    dual_metrics = binary_metrics(dual_labels, dual_probs)

    print(f"\n{'='*60}")
    print(f"BiomedCLIP Zero-Shot on CBIS-DDSM Test Set")
    print(f"{'='*60}")
    print(f"{'Metric':<10} {'CC-only':>10} {'Dual-view':>10}")
    print(f"{'-'*30}")
    for k in ['auroc', 'auprc', 'f1', 'accuracy', 'sensitivity', 'specificity', 'ece']:
        print(f"{k:<10} {cc_metrics[k]:10.4f} {dual_metrics[k]:10.4f}")

    save_json({"cc_only": cc_metrics, "dual_view": dual_metrics}, out_dir / "metrics.json")
    print(f"\nSaved to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
