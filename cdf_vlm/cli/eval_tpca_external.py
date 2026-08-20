"""Direct (zero-shot) TPCA inference on an external dataset (e.g. INbreast).

Loads the best CBIS-DDSM-trained TPCA checkpoint (BiomedCLIP backbone,
aligner + bilateral fusion + text prompts + consistency) and evaluates it
without any fine-tuning on an external paired-view manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cdf_vlm.biomedclip import load_local_biomedclip
from cdf_vlm.io import ensure_dir
from cdf_vlm.metrics import binary_metrics
from cdf_vlm.multiview_v2 import DDSMGeometryPromptModelV2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument(
        "--checkpoint",
        default="outputs/ddsm_ablation_v2/biomedclip_v2_with_consistency/model.pt",
    )
    p.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--output-dir", default="outputs/tpca_inbreast_eval")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DDSMGeometryPromptModelV2(
        clip_model=clip_model,
        freeze_backbone=True,
        use_aligner=True,
        use_bilateral=True,
        use_text_prompts=True,
        use_consistency=True,
    ).to(device)
    model.load_state_dict(ckpt["pair_model_state"], strict=False)
    model.eval()

    table = pd.read_csv(args.pair_csv)
    print(f"External pairs: {len(table)}")

    labels_all, probs_all = [], []
    for _, row in tqdm(table.iterrows(), total=len(table), desc="TPCA external eval"):
        try:
            cc_img = Image.open(f"{args.data_root}/{row['cc_image_path']}").convert("RGB")
            mlo_img = Image.open(f"{args.data_root}/{row['mlo_image_path']}").convert("RGB")
        except Exception as exc:
            print(f"  [WARN] cannot load {row['side_id']}: {exc}", file=sys.stderr)
            continue

        cc_t = preprocess(cc_img).unsqueeze(0).to(device)
        mlo_t = preprocess(mlo_img).unsqueeze(0).to(device)
        cc_mask = torch.ones(1, 1, cc_t.shape[-2], cc_t.shape[-1], device=device)
        mlo_mask = torch.ones(1, 1, mlo_t.shape[-2], mlo_t.shape[-1], device=device)

        out = model(
            cc_image=cc_t, cc_mask=cc_mask, cc_description=[row["cc_description"]],
            mlo_image=mlo_t, mlo_mask=mlo_mask, mlo_description=[row["mlo_description"]],
            side_ids=[row["side_id"]],
        )
        prob = torch.sigmoid(out["logits"]).cpu().item()
        labels_all.append(float(row["label"]))
        probs_all.append(prob)

    results = {"dual_view": binary_metrics(labels_all, probs_all)}
    out_dir = ensure_dir(args.output_dir)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"side_id": table["side_id"], "label": labels_all, "prob": probs_all}).to_csv(
        out_dir / "predictions.csv", index=False
    )

    m = results["dual_view"]
    print("\n=== TPCA (DDSM-trained) direct inference on external set ===")
    print(
        f"AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  F1={m['f1']:.4f}  "
        f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  ECE={m['ece']:.4f}"
    )
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
