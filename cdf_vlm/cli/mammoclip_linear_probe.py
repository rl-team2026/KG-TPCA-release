"""Mammo-CLIP linear probe on CBIS-DDSM (official downstream protocol).

Freeze the Mammo-CLIP image encoder, extract 512-d features, and train a
logistic regression classifier (the same protocol as the Mammo-CLIP paper's
linear-probe evaluations). Reports CC-only / MLO-only / dual-view (concat).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cdf_vlm.extractors.mammoclip import MammoCLIPImageExtractor
from cdf_vlm.io import ensure_dir
from cdf_vlm.metrics import binary_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODEBASE = PROJECT_ROOT / "external" / "Mammo-CLIP-src" / "src" / "codebase"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pair-csv", required=True)
    parser.add_argument("--val-pair-csv", required=True)
    parser.add_argument("--test-pair-csv", required=True)
    parser.add_argument("--data-root", default="data/processed/dcm1520x912")
    parser.add_argument("--train-data-root", default="")
    parser.add_argument("--val-data-root", default="")
    parser.add_argument("--test-data-root", default="")
    parser.add_argument(
        "--checkpoint",
        default="external/Mammo-CLIP/Pre-trained-checkpoints/b2-model-best-epoch-10.tar",
    )
    parser.add_argument("--source-dir", default=str(CODEBASE))
    parser.add_argument("--output-dir", default="outputs/mammoclip_linear_probe")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def extract_features(extractor, table: pd.DataFrame, data_root: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from cdf_vlm.extractors.mammoclip import build_mammo_preprocess

    preprocess = build_mammo_preprocess(extractor.config)
    device = extractor.device

    rows = []
    for _, row in tqdm(table.iterrows(), total=len(table), desc="loading"):
        try:
            cc = Image.open(f"{data_root}/{row['cc_image_path']}").convert("RGB")
            mlo = Image.open(f"{data_root}/{row['mlo_image_path']}").convert("RGB")
        except Exception as exc:
            print(f"  [WARN] cannot load {row['side_id']}: {exc}", file=sys.stderr)
            continue
        rows.append((float(row["label"]), cc, mlo))

    labels, feats_cc, feats_mlo = [], [], []
    for start in tqdm(range(0, len(rows), batch_size), desc="extracting"):
        chunk = rows[start : start + batch_size]
        labels.extend(r[0] for r in chunk)
        for col_idx in (1, 2):
            tensors = torch.stack([preprocess(r[col_idx]) for r in chunk]).to(device)
            feats = extractor.encode(tensors).cpu().numpy()
            (feats_cc if col_idx == 1 else feats_mlo).append(feats)

    return (
        np.asarray(labels),
        np.concatenate(feats_cc, axis=0),
        np.concatenate(feats_mlo, axis=0),
    )


def probe(train_feats: np.ndarray, train_labels: np.ndarray, test_feats: np.ndarray, test_labels: np.ndarray, val_feats=None, val_labels=None) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_feats)
    X_train = scaler.transform(train_feats)
    X_test = scaler.transform(test_feats)

    best_c, best_auc = 1.0, -1.0
    if val_feats is not None:
        X_val = scaler.transform(val_feats)
        for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(C=c, max_iter=2000).fit(X_train, train_labels)
            auc = roc_auc_score(val_labels, clf.predict_proba(X_val)[:, 1])
            if auc > best_auc:
                best_c, best_auc = c, auc
        print(f"  best C={best_c} (val AUROC={best_auc:.4f})")

    clf = LogisticRegression(C=best_c, max_iter=2000).fit(X_train, train_labels)
    probs = clf.predict_proba(X_test)[:, 1]
    return binary_metrics(test_labels.tolist(), probs.tolist())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    extractor = MammoCLIPImageExtractor(
        checkpoint=args.checkpoint, source_dir=args.source_dir, device=device,
    )

    print("\n--- train set ---")
    tr = extract_features(extractor, pd.read_csv(args.train_pair_csv), args.train_data_root or args.data_root, args.batch_size)
    print("--- val set ---")
    va = extract_features(extractor, pd.read_csv(args.val_pair_csv), args.val_data_root or args.data_root, args.batch_size)
    print("--- test set ---")
    te = extract_features(extractor, pd.read_csv(args.test_pair_csv), args.test_data_root or args.data_root, args.batch_size)

    results = {}
    for name, feat_idx in [("cc_only", 1), ("mlo_only", 2)]:
        print(f"\n=== linear probe: {name} ===")
        results[name] = probe(tr[feat_idx], tr[0], te[feat_idx], te[0], va[feat_idx], va[0])
    print("\n=== linear probe: dual_view (concat) ===")
    results["dual_view"] = probe(
        np.concatenate([tr[1], tr[2]], axis=1), tr[0],
        np.concatenate([te[1], te[2]], axis=1), te[0],
        np.concatenate([va[1], va[2]], axis=1), va[0],
    )

    out_dir = ensure_dir(args.output_dir)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    for name in ["train", "val", "test"]:
        feats = {"train": tr, "val": va, "test": te}[name]
        np.savez_compressed(out_dir / f"{name}_features.npz", labels=feats[0], cc=feats[1], mlo=feats[2])

    print("\n=== Mammo-CLIP linear probe on CBIS-DDSM ===")
    print(f"{'view':<12} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Sens':>7} {'Spec':>7} {'ECE':>7}")
    for name, m in results.items():
        print(
            f"{name:<12} {m['auroc']:>7.4f} {m['auprc']:>7.4f} {m['f1']:>7.4f} "
            f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f} {m['ece']:>7.4f}"
        )
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
