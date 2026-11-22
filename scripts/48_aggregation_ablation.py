#!/usr/bin/env python3
"""Aggregation ablation for the AAMAS rebuttal (Reviewer 1, Eq. 1 concern).

Compares three keyword aggregation mechanisms under the identical training
protocol used for the paper's Table 3/4 numbers (same data, same frozen
BiomedCLIP backbone, same optimizer and early stopping):
  linear    - normalized weighted average (paper Eq. 1)
  attention - softmax attention pooling over present keywords
  bilinear  - linear aggregate plus a low-rank pairwise interaction term
Each aggregation can be run with or without the BI-RADS KG regularization.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train_and_eval(config_name: str, aggregation: str, use_kg: bool, args, out_dir: Path) -> dict:
    import gc
    import itertools

    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from cdf_vlm.aggregations import AggPromptEncoder
    from cdf_vlm.biomedclip import load_local_biomedclip
    from cdf_vlm.config import save_json
    from cdf_vlm.ddsm import DDSMPairedViewDataset, DDSMSingleViewDataset
    from cdf_vlm.io import ensure_dir, seed_everything
    from cdf_vlm.metrics import binary_metrics
    from cdf_vlm.multiview_v2 import PromptAttentionGate

    seed_everything(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = ensure_dir(out_dir)
    print(f"Training: {config_name} (aggregation={aggregation}, use_kg={use_kg})")

    clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    from cdf_vlm.text_prompts import DEFAULT_KEYWORDS
    embed_dim = 512
    prompt_encoder = AggPromptEncoder(
        embed_dim=embed_dim,
        keywords=DEFAULT_KEYWORDS,
        aggregation=aggregation,
        use_kg=use_kg,
        kg_lambda_prior=args.kg_prior,
        kg_lambda_pairwise=args.kg_pairwise,
    )
    kg_loss_fn = prompt_encoder.compute_kg_loss if use_kg else None
    prompt_gate = PromptAttentionGate(embed_dim=embed_dim)

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
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable}")

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

            s_logits = model.forward_single(
                s_batch["image"].to(device),
                list(s_batch["description"]),
            )
            s_loss = criterion(s_logits, s_batch["label"].to(device))

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
            if kg_loss_fn is not None:
                kg_loss, _ = kg_loss_fn()
                loss = loss + kg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            del s_logits, cc_logits, mlo_logits, s_loss, p_loss, loss

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
        gc.collect()

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

    test_metrics["trainable_params"] = n_trainable
    save_json(test_metrics, out_dir / "metrics.json")
    torch.save(model.state_dict(), out_dir / "model.pt")

    print(f"{config_name} test: AUROC={test_metrics.get('auroc',0):.4f} "
          f"AUPRC={test_metrics.get('auprc',0):.4f} F1={test_metrics.get('f1',0):.4f} "
          f"ECE={test_metrics.get('ece',0):.4f}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Keyword aggregation ablation.")
    parser.add_argument("--single-csv", default="data/processed/ddsm/breast_cancer_train.csv")
    parser.add_argument("--pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_train.csv")
    parser.add_argument("--val-pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_val.csv")
    parser.add_argument("--test-pair-csv", default="data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv")
    parser.add_argument("--data-root", default="data/processed/ddsm")
    parser.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--output-dir", default="outputs/aggregation_ablation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--kg-prior", type=float, default=0.01)
    parser.add_argument("--kg-pairwise", type=float, default=0.005)
    parser.add_argument("--device")
    parser.add_argument("--runs", nargs="+", default=["attention_kg", "attention_nokg", "bilinear_kg", "bilinear_nokg"],
                        choices=["attention_kg", "attention_nokg", "bilinear_kg", "bilinear_nokg"],
                        help="Which variants to evaluate")
    args = parser.parse_args()

    results = {}
    for run_name in args.runs:
        aggregation, _, kg_flag = run_name.partition("_")
        use_kg = (kg_flag == "kg")
        out_dir = Path(args.output_dir) / run_name
        if (out_dir / "metrics.json").exists():
            print(f"skip {run_name} (metrics.json exists)")
            continue
        try:
            results[run_name] = train_and_eval(run_name, aggregation, use_kg, args, out_dir)
        except Exception as exc:
            print(f"[ERROR] {run_name} failed: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("AGGREGATION ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Variant':<16} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'ECE':>8}")
    for name, m in results.items():
        print(f"{name:<16} {m.get('auroc',0):>8.4f} {m.get('auprc',0):>8.4f} {m.get('f1',0):>8.4f} {m.get('ece',0):>8.4f}")


if __name__ == "__main__":
    main()
