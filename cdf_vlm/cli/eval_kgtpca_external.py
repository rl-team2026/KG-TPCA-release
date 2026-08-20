"""Direct (zero-shot) KG-TPCA inference on an external dataset (INbreast).

Replicates the KGConsistencyModel from scripts/run_kg_consistency_v2.py:
frozen BiomedCLIP + KG prompt encoder + prompt gate + MLP classifier,
dual-view = equal-weight feature average. Loads the best CBIS-DDSM
checkpoint (outputs/encoder_ablation/kg_enhanced_consistency_v3/model.pt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cdf_vlm.biomedclip import load_local_biomedclip
from cdf_vlm.birads_kg import KGEnhancedPromptEncoder
from cdf_vlm.io import ensure_dir
from cdf_vlm.metrics import binary_metrics
from cdf_vlm.multiview_v2 import PromptAttentionGate
from cdf_vlm.text_prompts import DEFAULT_KEYWORDS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default="outputs/encoder_ablation/kg_enhanced_consistency_v3/model.pt")
    p.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--output-dir", default="outputs/kgtpca_inbreast_eval")
    p.add_argument("--device", default="")
    return p.parse_args()


class KGConsistencyModel(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_encoder = KGEnhancedPromptEncoder(
            embed_dim=512, keywords=DEFAULT_KEYWORDS,
            kg_lambda_prior=0.01, kg_lambda_pairwise=0.005,
        )
        self.prompt_gate = PromptAttentionGate(embed_dim=512)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward_with_prompt(self, image: torch.Tensor, description: str) -> torch.Tensor:
        feat = self.clip_model.encode_image(image)
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        prompt = self.prompt_encoder.encode_texts([description], device=image.device)
        return self.prompt_gate(feat, prompt)

    def classify(self, feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(feat).squeeze(-1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    model = KGConsistencyModel(clip_model).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {len(state)} tensors; missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    table = pd.read_csv(args.pair_csv)
    print(f"External pairs: {len(table)}")

    labels_all, probs_all = [], []
    for _, row in tqdm(table.iterrows(), total=len(table), desc="KG-TPCA external eval"):
        try:
            cc_img = Image.open(f"{args.data_root}/{row['cc_image_path']}").convert("RGB")
            mlo_img = Image.open(f"{args.data_root}/{row['mlo_image_path']}").convert("RGB")
        except Exception as exc:
            print(f"  [WARN] cannot load {row['side_id']}: {exc}", file=sys.stderr)
            continue

        cc_t = preprocess(cc_img).unsqueeze(0).to(device)
        mlo_t = preprocess(mlo_img).unsqueeze(0).to(device)

        cc_feat = model.forward_with_prompt(cc_t, str(row["cc_description"]))
        mlo_feat = model.forward_with_prompt(mlo_t, str(row["mlo_description"]))
        logits = model.classify(0.5 * (cc_feat + mlo_feat))
        probs_all.append(torch.sigmoid(logits).cpu().item())
        labels_all.append(float(row["label"]))

    results = {"dual_view": binary_metrics(labels_all, probs_all)}
    out_dir = ensure_dir(args.output_dir)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"side_id": table["side_id"], "label": labels_all, "prob": probs_all}).to_csv(
        out_dir / "predictions.csv", index=False
    )

    m = results["dual_view"]
    print("\n=== KG-TPCA (DDSM-trained) direct inference on external set ===")
    print(
        f"AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  F1={m['f1']:.4f}  "
        f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  ECE={m['ece']:.4f}"
    )
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
