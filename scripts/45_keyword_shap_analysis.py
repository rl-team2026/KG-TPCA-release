#!/usr/bin/env python3
"""SHAP-based keyword contribution analysis for TPCA text prompt module.

Quantifies how each of the 16 pathology keywords contributes to the model's
malignancy prediction. Uses permutation feature importance on the multi-hot
keyword vector.

Usage:
  python scripts/45_keyword_shap_analysis.py \
    --checkpoint outputs/ddsm_ablation_v2/biomedclip_v2_baseline_prompt_only/model.pt \
    --test-csv data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv \
    --data-root data/processed/ddsm \
    --output-dir paper/figures/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    parser = argparse.ArgumentParser(description="Keyword SHAP analysis for TPCA.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--output-dir", default="paper/figures/analysis")
    parser.add_argument("--device")
    parser.add_argument("--n-samples", type=int, default=300)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from cdf_vlm.biomedclip import load_local_biomedclip
    from cdf_vlm.ddsm import DDSMPairedViewDataset
    from cdf_vlm.multiview_v2 import DDSMGeometryPromptModelV2
    from cdf_vlm.text_prompts import DEFAULT_KEYWORDS, keyword_multihot, extract_pathology_keywords

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pair_model = DDSMGeometryPromptModelV2(
        clip_model=clip_model, freeze_backbone=True,
        use_aligner=False, use_bilateral=False,
        use_text_prompts=True, use_consistency=False,
    ).to(device)
    pair_model.load_state_dict(ckpt["pair_model_state"], strict=False)
    pair_model.eval()

    test_dataset = DDSMPairedViewDataset(args.test_csv, data_root=args.data_root, image_transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)

    # ── Keyword-level permutation importance ──
    keywords = DEFAULT_KEYWORDS
    kw_deltas = {kw: [] for kw in keywords}
    kw_present_count = {kw: 0 for kw in keywords}

    print(f"Running keyword permutation analysis on up to {args.n_samples} samples...")
    sample_count = 0
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= args.n_samples:
                break
            for i in range(len(batch["label"])):
                if sample_count >= args.n_samples:
                    break
                cc_desc = batch["cc_description"][i]
                mlo_desc = batch["mlo_description"][i]
                cc_img = batch["cc_image"][i:i+1].to(device)
                cc_mask = batch["cc_mask"][i:i+1].to(device)
                mlo_img = batch["mlo_image"][i:i+1].to(device)
                mlo_mask = batch["mlo_mask"][i:i+1].to(device)
                side_id = [batch["side_id"][i]]

                # Baseline prediction
                out_base = pair_model(
                    cc_image=cc_img, cc_mask=cc_mask, cc_description=[cc_desc],
                    mlo_image=mlo_img, mlo_mask=mlo_mask, mlo_description=[mlo_desc],
                    side_ids=side_id,
                )
                prob_base = float(torch.sigmoid(out_base["logits"]).cpu().item())

                present_kws = set()
                for kw in keywords:
                    if kw in extract_pathology_keywords(cc_desc + " " + mlo_desc, [kw]):
                        present_kws.add(kw)

                for kw in present_kws:
                    # Remove keyword from description
                    cc_ablated = _remove_keyword(cc_desc, kw)
                    mlo_ablated = _remove_keyword(mlo_desc, kw)
                    if cc_ablated == cc_desc and mlo_ablated == mlo_desc:
                        continue

                    out_ablated = pair_model(
                        cc_image=cc_img, cc_mask=cc_mask, cc_description=[cc_ablated],
                        mlo_image=mlo_img, mlo_mask=mlo_mask, mlo_description=[mlo_ablated],
                        side_ids=side_id,
                    )
                    prob_ablated = float(torch.sigmoid(out_ablated["logits"]).cpu().item())
                    kw_deltas[kw].append(prob_base - prob_ablated)
                    kw_present_count[kw] += 1

                sample_count += 1
                if sample_count % 50 == 0:
                    print(f"  Processed {sample_count}/{args.n_samples} samples")

    # ── Summarize ──
    summary = {}
    for kw in keywords:
        deltas = kw_deltas[kw]
        if len(deltas) >= 3:
            arr = np.array(deltas)
            summary[kw] = {
                "mean_delta": float(np.mean(arr)),
                "std_delta": float(np.std(arr)),
                "abs_mean": float(np.mean(np.abs(arr))),
                "direction": "malignant" if np.mean(arr) > 0 else "benign",
                "n": len(deltas),
            }

    # Sort by absolute impact
    sorted_kws = sorted(summary.items(), key=lambda x: abs(x[1]["mean_delta"]), reverse=True)

    print("\nKeyword Contribution Summary (sorted by |impact|):")
    print(f"{'Keyword':<30} | {'Mean Δ':>7} | {'Std':>7} | {'N':>5} | {'Direction':>10}")
    print("-" * 72)
    for kw, s in sorted_kws:
        print(f"{kw:<30} | {s['mean_delta']:+7.4f} | {s['std_delta']:7.4f} | {s['n']:5d} | {s['direction']:>10}")

    # ── Generate figure ──
    n_plot = min(12, len(sorted_kws))
    plot_kws = sorted_kws[:n_plot]
    kw_names = [f"{kw}\n(n={s['n']})" for kw, s in plot_kws]
    mean_vals = [s["mean_delta"] for _, s in plot_kws]
    errors = [s["std_delta"] for _, s in plot_kws]
    colors = ["#F44336" if v > 0 else "#2196F3" for v in mean_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(kw_names))
    bars = ax.barh(y_pos, mean_vals, xerr=errors, color=colors, alpha=0.85,
                   edgecolor="white", capsize=3, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(kw_names, fontsize=9)
    ax.set_xlabel("Mean Prediction Change (Δ Probability)")
    ax.set_title("Pathology Keyword Contribution to Malignancy Prediction\n"
                 "(Positive = promotes malignant prediction, Negative = promotes benign prediction)",
                 fontweight="bold", fontsize=11)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="x", alpha=0.2)

    for bar, val in zip(bars, mean_vals):
        ax.text(val + 0.003 * np.sign(val), bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=8.5, fontweight="bold" if abs(val) > 0.05 else "normal")

    plt.tight_layout()
    fig.savefig(out_dir / "keyword_contribution.png")
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'keyword_contribution.png'}")

    # ── Grouped analysis: lesion type keywords vs localization keywords ──
    lesion_kws = ["mass", "calcification", "irregular", "spiculated", "circumscribed",
                  "obscured", "microlobulated", "architectural distortion"]
    spatial_kws = ["clustered", "pleomorphic", "linear", "segmental"]
    location_kws = ["left breast", "right breast", "cc view", "mlo view"]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    categories = {"Lesion Morphology": lesion_kws, "Spatial Distribution": spatial_kws, "Location/View": location_kws}
    cat_means = {}
    for cat_name, cat_kws in categories.items():
        vals = [summary[kw]["abs_mean"] for kw in cat_kws if kw in summary]
        cat_means[cat_name] = np.mean(vals) if vals else 0

    ax2.bar(cat_means.keys(), cat_means.values(), color=["#E91E63", "#9C27B0", "#607D8B"],
            alpha=0.85, edgecolor="white")
    ax2.set_ylabel("Mean |Δ Probability|")
    ax2.set_title("Keyword Contribution by Semantic Category", fontweight="bold")
    for i, (cat, val) in enumerate(cat_means.items()):
        ax2.text(i, val + 0.002, f"{val:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    fig2.savefig(out_dir / "keyword_category_contribution.png")
    plt.close(fig2)
    print(f"Saved: {out_dir / 'keyword_category_contribution.png'}")

    # Save numerical results
    (out_dir / "keyword_contributions.json").write_text(
        json.dumps({kw: s for kw, s in sorted_kws}, indent=2)
    )

    print("\nDone.")


def _remove_keyword(text: str, keyword: str) -> str:
    import re
    aliases = {
        "mass": r"\bmass\b",
        "calcification": r"\bcalcification\w*\b",
        "irregular": r"\birregular\b",
        "spiculated": r"\bspiculated\b",
        "circumscribed": r"\bcircumscribed\b",
        "obscured": r"\bobscured\b",
        "microlobulated": r"\bmicrolobulated\b",
        "architectural distortion": r"\barchitectural\s+distortion\b",
        "clustered": r"\bclustered\b",
        "pleomorphic": r"\bpleomorphic\b",
        "linear": r"\blinear\b",
        "segmental": r"\bsegmental\b",
        "left breast": r"\bleft\s+breast\b",
        "right breast": r"\bright\s+breast\b",
        "cc view": r"\bcc\s+view\b",
        "mlo view": r"\bmlo\s+view\b",
    }
    pattern = aliases.get(keyword, re.escape(keyword))
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


if __name__ == "__main__":
    main()
