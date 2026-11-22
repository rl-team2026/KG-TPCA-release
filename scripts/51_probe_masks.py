#!/usr/bin/env python3
"""Image+mask linear probe baselines for the AAMAS rebuttal (Reviewer 2, input fairness).

Extracts frozen BiomedCLIP features for the CBIS-DDSM pair splits and fits
logistic regression on two input conditions:
  global+local - dual-view average of global features concatenated with the
                 dual-view average of mask-enhanced (local) features
  global_only  - dual-view average of global features only
No text descriptions, no prompt modules. Uses the same splits as the paper's
Table 3/4 protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_train.csv")
    parser.add_argument("--val-pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_val.csv")
    parser.add_argument("--test-pair-csv", default="data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv")
    parser.add_argument("--data-root", default="data/processed/ddsm")
    parser.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--output-dir", default="outputs/probe_masks")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from cdf_vlm.biomedclip import load_local_biomedclip
    from cdf_vlm.config import save_json
    from cdf_vlm.ddsm import DDSMPairedViewDataset
    from cdf_vlm.io import ensure_dir, seed_everything
    from cdf_vlm.metrics import binary_metrics

    seed_everything(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = ensure_dir(args.output_dir)

    clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    def encode(img):
        with torch.no_grad():
            f = clip_model.encode_image(img.to(device))
        return F.normalize(f.float(), dim=-1)

    def masked(image, mask):
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode="nearest")
        return image * mask

    def extract(loader):
        feats_g, feats_l, labels = [], [], []
        for batch in loader:
            cc_g = encode(batch["cc_image"])
            mlo_g = encode(batch["mlo_image"])
            cc_l = encode(masked(batch["cc_image"], batch["mask_cc"] if "mask_cc" in batch else batch["cc_mask"]))
            mlo_l = encode(masked(batch["mlo_image"], batch["mask_mlo"] if "mask_mlo" in batch else batch["mlo_mask"]))
            feats_g.append(((cc_g + mlo_g) * 0.5).cpu())
            feats_l.append(((cc_l + mlo_l) * 0.5).cpu())
            labels.append(batch["label"].cpu())
        return (
            torch.cat(feats_g).numpy(),
            torch.cat(feats_l).numpy(),
            torch.cat(labels).numpy(),
        )

    train_ds = DDSMPairedViewDataset(args.pair_csv, data_root=args.data_root, image_transform=preprocess)
    val_ds = DDSMPairedViewDataset(args.val_pair_csv, data_root=args.data_root, image_transform=preprocess)
    test_ds = DDSMPairedViewDataset(args.test_pair_csv, data_root=args.data_root, image_transform=preprocess)
    tr_g, tr_l, tr_y = extract(DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers))
    va_g, va_l, va_y = extract(DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers))
    te_g, te_l, te_y = extract(DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers))
    print("features extracted")

    from sklearn.linear_model import LogisticRegression

    def eval_lr(X_tr, X_va, X_te, name):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X_tr, tr_y)
        probs = clf.predict_proba(X_te)[:, 1]
        m = binary_metrics(te_y.tolist(), probs.tolist())
        m["val_auroc"] = float(binary_metrics(va_y.tolist(), clf.predict_proba(X_va)[:, 1].tolist())["auroc"])
        save_json(m, out_dir / f"{name}.json")
        print(f"{name}: test AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} F1={m['f1']:.4f} ECE={m['ece']:.4f}")
        return m

    eval_lr(tr_g, va_g, te_g, "probe_global_only")
    eval_lr(np.concatenate([tr_g, tr_l], axis=1),
            np.concatenate([va_g, va_l], axis=1),
            np.concatenate([te_g, te_l], axis=1), "probe_global_local")


if __name__ == "__main__":
    main()
