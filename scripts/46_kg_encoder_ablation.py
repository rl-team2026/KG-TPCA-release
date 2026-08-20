#!/usr/bin/env python3
"""Compare KG-enhanced text prompt encoder vs 16-keyword baseline on CBIS-DDSM.

Runs 3 configurations:
  A. Baseline: PathologyPromptEncoder (16-keyword multi-hot → MLP)
  B. KG-Enhanced: KGEnhancedPromptEncoder (learned keyword embeddings + KG loss)
  C. Structured: StructuredFieldPromptEncoder (243-dim structured fields → MLP)

All use frozen BiomedCLIP backbone, no aligner/fusion (prompt-only config).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def train_and_eval(config_name: str, encoder_type: str, args, out_dir: Path) -> dict:
    """Train one configuration and return test metrics."""
    import gc
    import itertools

    import numpy as np
    import pandas as pd
    import psutil
    import torch
    from tqdm import tqdm
    from torch import nn
    from torch.utils.data import DataLoader

    from cdf_vlm.biomedclip import load_local_biomedclip
    from cdf_vlm.config import save_json
    from cdf_vlm.ddsm import DDSMPairedViewDataset, DDSMSingleViewDataset
    from cdf_vlm.io import ensure_dir, seed_everything
    from cdf_vlm.metrics import binary_metrics
    from cdf_vlm.multiview_v2 import PromptAttentionGate

    seed_everything(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = ensure_dir(out_dir)
    print(f"\n{'='*60}")
    print(f"Training: {config_name} ({encoder_type})")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    # Load BiomedCLIP
    clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    # Build encoder
    from cdf_vlm.text_prompts import DEFAULT_KEYWORDS
    embed_dim = 512
    kg_loss_fn = None

    if encoder_type == "baseline":
        from cdf_vlm.text_prompts import PathologyPromptEncoder
        prompt_encoder = PathologyPromptEncoder(embed_dim=embed_dim, keywords=DEFAULT_KEYWORDS)
    elif encoder_type == "kg_enhanced":
        from cdf_vlm.birads_kg import KGEnhancedPromptEncoder
        prompt_encoder = KGEnhancedPromptEncoder(
            embed_dim=embed_dim, keywords=DEFAULT_KEYWORDS,
            kg_lambda_prior=args.kg_prior, kg_lambda_pairwise=args.kg_pairwise,
        )
        kg_loss_fn = prompt_encoder.compute_kg_loss
    elif encoder_type == "structured":
        from cdf_vlm.structured_prompts import StructuredFieldPromptEncoder
        prompt_encoder = StructuredFieldPromptEncoder(embed_dim=embed_dim, per_field_dim=32)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    prompt_gate = PromptAttentionGate(embed_dim=embed_dim)

    # Single-view model (minimal: no aligner/fusion)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_encoder = prompt_encoder
            self.prompt_gate = prompt_gate
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
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

    model = SimpleModel().to(device)

    # Data
    single_dataset = DDSMSingleViewDataset(args.single_csv, data_root=args.data_root, image_transform=preprocess)
    pair_dataset = DDSMPairedViewDataset(args.pair_csv, data_root=args.data_root, image_transform=preprocess)
    single_loader = DataLoader(single_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    pair_loader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = None
    if args.val_pair_csv:
        val_dataset = DDSMPairedViewDataset(args.val_pair_csv, data_root=args.data_root, image_transform=preprocess)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = None
    if args.test_pair_csv:
        test_dataset = DDSMPairedViewDataset(args.test_pair_csv, data_root=args.data_root, image_transform=preprocess)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
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

            # Single-view: use CC description
            s_logits = model.forward_single(
                s_batch["image"].to(device),
                list(s_batch["description"]),
            )
            s_loss = criterion(s_logits, s_batch["label"].to(device))

            # Pair-view: simple average of CC + MLO
            cc_logits = model.forward_single(
                p_batch["cc_image"].to(device),
                list(p_batch["cc_description"]),
            )
            mlo_logits = model.forward_single(
                p_batch["mlo_image"].to(device),
                list(p_batch["mlo_description"]),
            )
            p_loss = criterion((cc_logits + mlo_logits) * 0.5, p_batch["label"].to(device))

            loss = s_loss + p_loss

            # KG regularization
            if kg_loss_fn is not None:
                kg_loss, _ = kg_loss_fn()
                loss = loss + kg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            del s_logits, cc_logits, mlo_logits, s_loss, p_loss, loss

        # Validation
        if val_loader is not None:
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
            if patience_counter >= args.patience and epoch > 3:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            print(f"Epoch {epoch:2d}: loss={np.mean(losses):.4f}")

    # Restore best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = {}
    if test_loader is not None:
        model.eval()
        probs_all, labels_all = [], []
        with torch.no_grad():
            for batch in test_loader:
                cc = model.forward_single(batch["cc_image"].to(device), list(batch["cc_description"]))
                mlo = model.forward_single(batch["mlo_image"].to(device), list(batch["mlo_description"]))
                probs = torch.sigmoid((cc + mlo) * 0.5).cpu().numpy()
                probs_all.extend(probs.tolist())
                labels_all.extend(batch["label"].cpu().numpy().tolist())
        test_metrics = binary_metrics(labels_all, probs_all)

    save_json(test_metrics, out_dir / "metrics.json")
    torch.save(model.state_dict(), out_dir / "model.pt")

    print(f"\n{config_name} test: AUROC={test_metrics.get('auroc',0):.4f} "
          f"AUPRC={test_metrics.get('auprc',0):.4f} F1={test_metrics.get('f1',0):.4f} "
          f"ECE={test_metrics.get('ece',0):.4f}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare text prompt encoder variants.")
    parser.add_argument("--single-csv", default="data/processed/ddsm/breast_cancer_train.csv")
    parser.add_argument("--pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_train.csv")
    parser.add_argument("--val-pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_val.csv")
    parser.add_argument("--test-pair-csv", default="data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv")
    parser.add_argument("--data-root", default="data/processed/ddsm")
    parser.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--output-dir", default="outputs/encoder_ablation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--kg-prior", type=float, default=0.01,
                        help="Lambda for BI-RADS prior loss")
    parser.add_argument("--kg-pairwise", type=float, default=0.005,
                        help="Lambda for BI-RADS pairwise relationship loss")
    parser.add_argument("--device")
    parser.add_argument("--encoders", nargs="+", default=["baseline", "kg_enhanced", "structured"],
                        choices=["baseline", "kg_enhanced", "structured"],
                        help="Which encoders to evaluate")
    args = parser.parse_args()

    results = {}
    for enc_type in args.encoders:
        name_map = {
            "baseline": "16-keyword multi-hot → MLP (Baseline)",
            "kg_enhanced": "KG-Enhanced embeddings + BI-RADS prior",
            "structured": "Structured field encoder (243-dim)",
        }
        config_name = enc_type
        human_name = name_map[enc_type]
        out_dir = Path(args.output_dir) / enc_type

        try:
            metrics = train_and_eval(config_name, enc_type, args, out_dir)
            results[human_name] = metrics
        except Exception as e:
            print(f"[ERROR] {config_name} failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TEXT PROMPT ENCODER ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Encoder':<45} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'ECE':>7}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<45} {m.get('auroc',0):7.4f} {m.get('auprc',0):7.4f} "
              f"{m.get('f1',0):7.4f} {m.get('ece',0):7.4f}")

    baseline_auroc = results.get(
        "16-keyword multi-hot → MLP (Baseline)", {}
    ).get("auroc", 0)
    for name, m in results.items():
        if name != "16-keyword multi-hot → MLP (Baseline)" and baseline_auroc > 0:
            delta = m.get("auroc", 0) - baseline_auroc
            print(f"  {name}: ΔAUROC = {delta:+.4f} vs baseline")

    # Save summary
    import json
    (Path(args.output_dir) / "encoder_ablation_summary.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
