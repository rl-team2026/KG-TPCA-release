#!/usr/bin/env python3
"""KG-TPCA + Cross-View Consistency Loss experiment.
Re-run to get consistent Acc/Sens/Spec metrics for the paper's SOTA table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cdf_vlm.biomedclip import load_local_biomedclip
from cdf_vlm.config import save_json
from cdf_vlm.ddsm import DDSMPairedViewDataset, DDSMSingleViewDataset
from cdf_vlm.io import ensure_dir, seed_everything
from cdf_vlm.metrics import binary_metrics
from cdf_vlm.multiview_v2 import PromptAttentionGate
from cdf_vlm.text_prompts import DEFAULT_KEYWORDS
from cdf_vlm.birads_kg import KGEnhancedPromptEncoder

# ═══════════════════════ CONFIG ═══════════════════════
DATA_ROOT = "data/processed/ddsm"
SINGLE_CSV = f"{DATA_ROOT}/breast_cancer_train.csv"
PAIR_CSV = f"{DATA_ROOT}/breast_cancer_train_cc_mlo_pair_train.csv"
VAL_PAIR_CSV = f"{DATA_ROOT}/breast_cancer_train_cc_mlo_pair_val.csv"
TEST_PAIR_CSV = f"{DATA_ROOT}/breast_cancer_test_cc_mlo_pair.csv"
BIOMEDCLIP_DIR = "external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OUT_DIR = Path("outputs/encoder_ablation/kg_enhanced_consistency")

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
NUM_WORKERS = 2
SEED = 42

# KG loss weights (from paper Section 1.2, Eq. 4)
KG_LAMBDA_PRIOR = 0.01
KG_LAMBDA_PAIRWISE = 0.005
# Consistency loss weight (from paper Section 1.5)
LAMBDA_CONSIST = 0.05


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_dir = ensure_dir(OUT_DIR)
    print(f"Output: {out_dir}")

    # Load BiomedCLIP (frozen)
    print("Loading BiomedCLIP...")
    clip_model, preprocess, _ = load_local_biomedclip(BIOMEDCLIP_DIR, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    # Build KG-enhanced prompt encoder
    prompt_encoder = KGEnhancedPromptEncoder(
        embed_dim=512, keywords=DEFAULT_KEYWORDS,
        kg_lambda_prior=KG_LAMBDA_PRIOR, kg_lambda_pairwise=KG_LAMBDA_PAIRWISE,
    )
    prompt_gate = PromptAttentionGate(embed_dim=512)

    class KGConsistencyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_encoder = prompt_encoder
            self.prompt_gate = prompt_gate
            self.classifier = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 1),
            )

        def encode_image(self, img):
            f = clip_model.encode_image(img)
            return torch.nn.functional.normalize(f.float(), dim=-1)

        def forward_single(self, image, description):
            feat = self.encode_image(image)
            prompt = self.prompt_encoder.encode_texts(description, device=image.device)
            feat = self.prompt_gate(feat, prompt)
            return self.classifier(feat).squeeze(-1)

    model = KGConsistencyModel().to(device)

    # Data
    single_dataset = DDSMSingleViewDataset(SINGLE_CSV, data_root=DATA_ROOT, image_transform=preprocess)
    pair_dataset = DDSMPairedViewDataset(PAIR_CSV, data_root=DATA_ROOT, image_transform=preprocess)
    single_loader = DataLoader(single_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    pair_loader = DataLoader(pair_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_dataset = DDSMPairedViewDataset(VAL_PAIR_CSV, data_root=DATA_ROOT, image_transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_dataset = DDSMPairedViewDataset(TEST_PAIR_CSV, data_root=DATA_ROOT, image_transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train single: {len(single_dataset)}, pair: {len(pair_dataset)}")
    print(f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        single_iter = iter(single_loader)
        pair_iter = iter(pair_loader)
        steps = max(len(pair_loader), len(single_loader))

        for _ in range(steps):
            try:
                s_batch = next(single_iter)
            except StopIteration:
                single_iter = iter(single_loader); s_batch = next(single_iter)
            try:
                p_batch = next(pair_iter)
            except StopIteration:
                pair_iter = iter(pair_loader); p_batch = next(pair_iter)

            # Single-view
            s_logits = model.forward_single(
                s_batch["image"].to(device),
                list(s_batch["description"]),
            )
            s_loss = criterion(s_logits, s_batch["label"].to(device))

            # Pair-view: CC and MLO separately
            cc_logits = model.forward_single(
                p_batch["cc_image"].to(device),
                list(p_batch["cc_description"]),
            )
            mlo_logits = model.forward_single(
                p_batch["mlo_image"].to(device),
                list(p_batch["mlo_description"]),
            )
            # Fused prediction
            fused_logits = (cc_logits + mlo_logits) * 0.5
            p_loss = criterion(fused_logits, p_batch["label"].to(device))

            # Cross-view consistency loss (from paper Eq. 6)
            # L_consist = ½[‖z_cc − sg(z_mlo)‖² + ‖z_mlo − sg(z_cc)‖²]
            consist_loss = 0.5 * (
                torch.nn.functional.mse_loss(cc_logits, mlo_logits.detach()) +
                torch.nn.functional.mse_loss(mlo_logits, cc_logits.detach())
            )

            loss = s_loss + p_loss + LAMBDA_CONSIST * consist_loss

            # KG regularization
            kg_loss, _ = prompt_encoder.compute_kg_loss()
            loss = loss + kg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            del s_logits, cc_logits, mlo_logits, s_loss, p_loss, loss

        # Validation
        model.eval()
        probs_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                cc = model.forward_single(batch["cc_image"].to(device), list(batch["cc_description"]))
                mlo = model.forward_single(batch["mlo_image"].to(device), list(batch["mlo_description"]))
                probs = torch.sigmoid((cc + mlo) * 0.5).cpu().numpy()
                probs_all.extend(probs.tolist())
                labels_all.extend(batch["label"].cpu().numpy().tolist())
        val_metrics = binary_metrics(labels_all, probs_all)
        val_auroc = val_metrics["auroc"]

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch:2d}: loss={np.mean(losses):.4f} val_auroc={val_auroc:.4f} best={best_auroc:.4f}")
        if patience_counter >= PATIENCE and epoch > 3:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            cc = model.forward_single(batch["cc_image"].to(device), list(batch["cc_description"]))
            mlo = model.forward_single(batch["mlo_image"].to(device), list(batch["mlo_description"]))
            probs = torch.sigmoid((cc + mlo) * 0.5).cpu().numpy()
            test_probs.extend(probs.tolist())
            test_labels.extend(batch["label"].cpu().numpy().tolist())

    test_metrics = binary_metrics(test_labels, test_probs)
    save_json(test_metrics, out_dir / "metrics.json")
    torch.save(model.state_dict(), out_dir / "model.pt")

    # Save predictions
    import pandas as pd
    pred_df = pd.DataFrame({"label": test_labels, "prob": test_probs})
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n{'='*60}")
    print(f"KG + Consistency Test Results")
    print(f"{'='*60}")
    print(f"AUROC:  {test_metrics['auroc']:.4f}")
    print(f"AUPRC:  {test_metrics['auprc']:.4f}")
    print(f"F1:     {test_metrics['f1']:.4f}")
    print(f"Acc:    {test_metrics['accuracy']:.4f}")
    print(f"Sens:   {test_metrics['sensitivity']:.4f}")
    print(f"Spec:   {test_metrics['specificity']:.4f}")
    print(f"ECE:    {test_metrics['ece']:.4f}")
    print(f"\nResults saved to {out_dir}/")
    return test_metrics


if __name__ == "__main__":
    main()
