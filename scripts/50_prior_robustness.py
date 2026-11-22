#!/usr/bin/env python3
"""Prior robustness and multi-seed experiments for the AAMAS rebuttal (Reviewer 2).

Runs under the identical protocol as the paper's Table 3/4 ablation script:
  - multi-seed repeats of the KG reference config and the no-KG baseline
  - shuffled malignancy prior scores (wrong-prior control)
  - noisy malignancy prior scores (uniform +/- 0.1)
  - shuffled semantic relation targets
  - no-text ablation (image-only pipeline, no prompt)

Run tags (--runs):
  seed123_kg, seed456_kg, seed123_base, seed456_base,
  shuffled_prior, noisy_prior, shuffled_rel, no_text
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train_and_eval(run_name: str, args, out_dir: Path) -> dict:
    import gc
    import itertools

    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from cdf_vlm.biomedclip import load_local_biomedclip
    from cdf_vlm.birads_kg import (
        BIRADSKnowledgeGraphLoss,
        KGEnhancedPromptEncoder,
        MALIGNANCY_PRIORS,
        PAIRWISE_RELATIONSHIPS,
    )
    from cdf_vlm.config import save_json
    from cdf_vlm.ddsm import DDSMPairedViewDataset, DDSMSingleViewDataset
    from cdf_vlm.io import ensure_dir, seed_everything
    from cdf_vlm.metrics import binary_metrics
    from cdf_vlm.multiview_v2 import PromptAttentionGate

    parts = run_name.split("_")
    seed = 42
    use_kg = True
    prior_mode = "clinical"
    relation_mode = "clinical"
    use_text = True

    if run_name.startswith("seed") and run_name.endswith("kg"):
        seed = int(run_name[4:-3])
    elif run_name.startswith("seed") and run_name.endswith("base"):
        seed = int(run_name[4:-5])
        use_kg = False
    elif run_name == "shuffled_prior":
        prior_mode = "shuffled"
    elif run_name == "noisy_prior":
        prior_mode = "noisy"
    elif run_name == "shuffled_rel":
        relation_mode = "shuffled"
    elif run_name == "no_text":
        use_kg = False
        use_text = False
    else:
        raise ValueError(f"Unknown run: {run_name}")

    seed_everything(seed)
    rng = np.random.RandomState(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = ensure_dir(out_dir)
    print(f"Training: {run_name} (seed={seed}, use_kg={use_kg}, prior={prior_mode}, "
          f"relation={relation_mode}, use_text={use_text})")

    clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    from cdf_vlm.text_prompts import DEFAULT_KEYWORDS
    embed_dim = 512
    encoder = KGEnhancedPromptEncoder(
        embed_dim=embed_dim,
        keywords=DEFAULT_KEYWORDS,
        kg_lambda_prior=args.kg_prior,
        kg_lambda_pairwise=args.kg_pairwise,
    )

    # Apply prior perturbations
    priors = dict(MALIGNANCY_PRIORS)
    relations = list(PAIRWISE_RELATIONSHIPS)
    if prior_mode == "shuffled":
        kw_list = list(priors.keys())
        vals = [priors[k] for k in kw_list]
        perm = rng.permutation(len(vals))
        priors = {k: vals[perm[i]] for i, k in enumerate(kw_list)}
    elif prior_mode == "noisy":
        priors = {k: float(np.clip(v + rng.uniform(-0.1, 0.1), -1.0, 1.0)) for k, v in priors.items()}
    if relation_mode == "shuffled":
        taus = [t for _, _, t in relations]
        perm = rng.permutation(len(taus))
        relations = [(a, b, taus[perm[i]]) for i, (a, b, _) in enumerate(relations)]

    kg_loss_fn = None
    if use_kg:
        custom_kg = BIRADSKnowledgeGraphLoss(
            keywords=DEFAULT_KEYWORDS,
            priors=priors,
            relationships=relations,
            lambda_prior=args.kg_prior,
            lambda_pairwise=args.kg_pairwise,
        )
        encoder.kg_loss_fn = custom_kg
        kg_loss_fn = encoder.compute_kg_loss

    prompt_gate = PromptAttentionGate(embed_dim=embed_dim)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_encoder = encoder
            self.prompt_gate = prompt_gate
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 1),
            )

        def encode_image(self, img):
            f = clip_model.encode_image(img)
            return torch.nn.functional.normalize(f.float(), dim=-1)

        def forward_single(self, image, description=None):
            feat = self.encode_image(image)
            if use_text:
                prompt = self.prompt_encoder.encode_texts(description, device=image.device)
                feat = self.prompt_gate(feat, prompt)
            return self.classifier(feat).squeeze(-1)

    model = SimpleModel().to(device)

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
                list(s_batch["description"]) if use_text else None,
            )
            s_loss = criterion(s_logits, s_batch["label"].to(device))

            cc_logits = model.forward_single(
                p_batch["cc_image"].to(device),
                list(p_batch["cc_description"]) if use_text else None,
            )
            mlo_logits = model.forward_single(
                p_batch["mlo_image"].to(device),
                list(p_batch["mlo_description"]) if use_text else None,
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
                    cc = model.forward_single(
                        batch["cc_image"].to(device),
                        list(batch["cc_description"]) if use_text else None,
                    )
                    mlo = model.forward_single(
                        batch["mlo_image"].to(device),
                        list(batch["mlo_description"]) if use_text else None,
                    )
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
                cc = model.forward_single(
                    batch["cc_image"].to(device),
                    list(batch["cc_description"]) if use_text else None,
                )
                mlo = model.forward_single(
                    batch["mlo_image"].to(device),
                    list(batch["mlo_description"]) if use_text else None,
                )
                probs = torch.sigmoid((cc + mlo) * 0.5).cpu().numpy()
                probs_all.extend(probs.tolist())
                labels_all.extend(batch["label"].cpu().numpy().tolist())
        test_metrics = binary_metrics(labels_all, probs_all)

    test_metrics["seed"] = seed
    test_metrics["prior_mode"] = prior_mode
    test_metrics["relation_mode"] = relation_mode
    test_metrics["use_kg"] = use_kg
    test_metrics["use_text"] = use_text
    save_json(test_metrics, out_dir / "metrics.json")
    torch.save(model.state_dict(), out_dir / "model.pt")

    print(f"{run_name} test: AUROC={test_metrics.get('auroc',0):.4f} "
          f"AUPRC={test_metrics.get('auprc',0):.4f} F1={test_metrics.get('f1',0):.4f} "
          f"ECE={test_metrics.get('ece',0):.4f}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Prior robustness and multi-seed experiments.")
    parser.add_argument("--single-csv", default="data/processed/ddsm/breast_cancer_train.csv")
    parser.add_argument("--pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_train.csv")
    parser.add_argument("--val-pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_val.csv")
    parser.add_argument("--test-pair-csv", default="data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv")
    parser.add_argument("--data-root", default="data/processed/ddsm")
    parser.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--output-dir", default="outputs/prior_robustness")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--kg-prior", type=float, default=0.01)
    parser.add_argument("--kg-pairwise", type=float, default=0.005)
    parser.add_argument("--device")
    parser.add_argument("--runs", nargs="+",
                        default=["seed123_kg", "seed456_kg", "seed123_base", "seed456_base",
                                 "shuffled_prior", "noisy_prior", "shuffled_rel", "no_text"])
    args = parser.parse_args()

    results = {}
    for run_name in args.runs:
        out_dir = Path(args.output_dir) / run_name
        if (out_dir / "metrics.json").exists():
            print(f"skip {run_name} (metrics.json exists)")
            continue
        try:
            results[run_name] = train_and_eval(run_name, args, out_dir)
        except Exception as exc:
            print(f"[ERROR] {run_name} failed: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("PRIOR ROBUSTNESS RESULTS")
    print("=" * 70)
    for name, m in results.items():
        print(f"{name:<16} AUROC={m.get('auroc',0):.4f} AUPRC={m.get('auprc',0):.4f} "
              f"F1={m.get('f1',0):.4f} ECE={m.get('ece',0):.4f}")


if __name__ == "__main__":
    main()
