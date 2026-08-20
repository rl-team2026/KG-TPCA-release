"""Mammo-CLIP zero-shot evaluation on CBIS-DDSM (official protocol).

Follows the official Mammo-CLIP zero-shot pipeline (evaluator.py):
  - prompts: ["no malignancy", "malignancy"] (RSNA-style, zs_clip.yaml)
  - text: Bio_ClinicalBERT, eos pooling -> text projection -> L2 norm
  - image: EfficientNet-B2 -> image projection -> L2 norm
  - score: softmax over cosine similarities (no temperature)

Reports CC-only / MLO-only / dual-view (mean prob) metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import softmax as scipy_softmax
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cdf_vlm.extractors.mammoclip import MammoCLIPImageExtractor
from cdf_vlm.io import ensure_dir
from cdf_vlm.metrics import binary_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODEBASE = PROJECT_ROOT / "external" / "Mammo-CLIP-src" / "src" / "codebase"
HF_ROOT = PROJECT_ROOT / "external" / "Mammo-CLIP" / "huggingface"

PROMPTS = ["no malignancy", "malignancy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--data-root", default="data/processed/dcm1520x912")
    parser.add_argument(
        "--checkpoint",
        default="external/Mammo-CLIP/Pre-trained-checkpoints/b2-model-best-epoch-10.tar",
    )
    parser.add_argument("--source-dir", default=str(CODEBASE))
    parser.add_argument("--output-dir", default="outputs/mammoclip_zeroshot")
    parser.add_argument("--input-size", type=int, nargs=2, default=[1520, 912])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def load_text_side(checkpoint_path: Path, device: torch.device):
    """Build Bio_ClinicalBERT + text projection from checkpoint weights."""
    from cdf_vlm.extractors.mammoclip import _load_modules

    _load_modules(CODEBASE)
    from breastclip.model.modules.projection import LinearProjectionHead

    from transformers import AutoConfig, BertModel, BertTokenizer

    tokenizer_snapshot = sorted(
        (HF_ROOT / "tokenizers" / "models--emilyalsentzer--Bio_ClinicalBERT" / "snapshots").iterdir()
    )[-1]
    model_snapshot = sorted(
        (HF_ROOT / "models--emilyalsentzer--Bio_ClinicalBERT" / "snapshots").iterdir()
    )[-1]

    tokenizer = BertTokenizer.from_pretrained(str(tokenizer_snapshot), local_files_only=True)
    bert_config = AutoConfig.from_pretrained(str(model_snapshot), local_files_only=True)
    bert = BertModel(bert_config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {
        key.replace("text_encoder.", ""): value
        for key, value in ckpt["model"].items()
        if key.startswith("text_encoder.")
    }
    missing, unexpected = bert.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] BERT missing keys: {missing[:5]}")

    text_projection = LinearProjectionHead(bert_config.hidden_size, 512)
    proj_state = {
        key.replace("text_projection.", ""): value
        for key, value in ckpt["model"].items()
        if key.startswith("text_projection.")
    }
    text_projection.load_state_dict(proj_state, strict=True)

    bert.to(device).eval()
    text_projection.to(device).eval()
    return tokenizer, bert, text_projection


@torch.no_grad()
def encode_texts(tokenizer, bert, text_projection, prompts: list[str], device: torch.device) -> np.ndarray:
    tokens = tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt", max_length=256)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    out = bert(**tokens)["last_hidden_state"]
    eos_indices = tokens["attention_mask"].sum(dim=-1) - 1
    feats = out[torch.arange(out.shape[0], device=device), eos_indices]
    feats = text_projection(feats)
    feats = feats / feats.norm(dim=1, keepdim=True)
    return feats.cpu().numpy()


def evaluate_views(image_extractor, text_emb: np.ndarray, table: pd.DataFrame, args) -> dict:
    from cdf_vlm.extractors.mammoclip import build_mammo_preprocess

    preprocess = build_mammo_preprocess(image_extractor.config)
    device = image_extractor.device

    rows = []
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Mammo-CLIP zero-shot"):
        try:
            cc_img = Image.open(args.data_root + "/" + row["cc_image_path"]).convert("RGB")
            mlo_img = Image.open(args.data_root + "/" + row["mlo_image_path"]).convert("RGB")
        except Exception as exc:
            print(f"  [WARN] cannot load {row['side_id']}: {exc}", file=sys.stderr)
            continue
        rows.append((row["label"], cc_img, mlo_img))

    labels, probs_cc, probs_mlo = [], [], []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="encoding"):
        chunk = rows[start : start + args.batch_size]
        labels.extend(float(r[0]) for r in chunk)
        for col_idx in (1, 2):
            tensors = torch.stack([preprocess(r[col_idx]) for r in chunk]).to(device)
            img_emb = image_extractor.encode(tensors).cpu().numpy()
            sims = scipy_softmax(img_emb @ text_emb.T, axis=1)
            (probs_cc if col_idx == 1 else probs_mlo).extend(sims[:, 1].tolist())

    probs_dual = [(c + m) / 2.0 for c, m in zip(probs_cc, probs_mlo)]
    return {
        "cc_only": binary_metrics(labels, probs_cc),
        "mlo_only": binary_metrics(labels, probs_mlo),
        "dual_view": binary_metrics(labels, probs_dual),
    }, {"label": labels, "prob_cc": probs_cc, "prob_mlo": probs_mlo, "prob_dual": probs_dual}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    image_extractor = MammoCLIPImageExtractor(
        checkpoint=args.checkpoint, source_dir=args.source_dir, device=device,
    )
    tokenizer, bert, text_projection = load_text_side(Path(args.checkpoint), device)
    text_emb = encode_texts(tokenizer, bert, text_projection, PROMPTS, device)
    print(f"Text embeddings: {text_emb.shape}")

    table = pd.read_csv(args.pair_csv)
    print(f"Test pairs: {len(table)}")

    results, preds = evaluate_views(image_extractor, text_emb, table, args)

    out_dir = ensure_dir(args.output_dir)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)

    print("\n=== Mammo-CLIP zero-shot on CBIS-DDSM ===")
    print(f"{'view':<12} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Sens':>7} {'Spec':>7} {'ECE':>7}")
    for name, m in results.items():
        print(
            f"{name:<12} {m['auroc']:>7.4f} {m['auprc']:>7.4f} {m['f1']:>7.4f} "
            f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f} {m['ece']:>7.4f}"
        )
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
