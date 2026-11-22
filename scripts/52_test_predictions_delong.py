#!/usr/bin/env python3
"""Collect test predictions from published checkpoints; run DeLong tests and bootstrap CIs.

Models (all evaluated on the same 266-pair CBIS-DDSM test CSV, deterministic order):
  kgtpca      outputs/encoder_ablation/kg_enhanced/model.pt        (KGEnhancedPromptEncoder)
  baseline    outputs/encoder_ablation/baseline/model.pt           (PathologyPromptEncoder)
  coop_cosine outputs/coop_baselines/coop_cosine/model.pt          (CoOpTextAdapter)
  coop_gate   outputs/coop_baselines/coop_gate/model.pt            (ContextPromptEncoder, no cond)
  cocoop_gate outputs/coop_baselines/cocoop_gate/model.pt          (ContextPromptEncoder, cond)
  no_text     outputs/prior_robustness/no_text/model.pt            (image-only)
  seed{123,456}_{kg,base}  outputs/prior_robustness/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def fast_delong_components(y_true: np.ndarray, scores: np.ndarray):
    """Per-observation structural components V10 (positives) / V01 (negatives) and AUC."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    m = int(y.sum())
    k = len(y) - m
    if m == 0 or k == 0:
        raise ValueError("need both classes")
    pos = y == 1
    neg = y == 0
    v10 = np.zeros(len(y))
    v01 = np.zeros(len(y))
    for i in np.where(pos)[0]:
        v10[i] = (np.sum(s[neg] < s[i]) + 0.5 * np.sum(s[neg] == s[i])) / k
    for j in np.where(neg)[0]:
        v01[j] = (np.sum(s[pos] > s[j]) + 0.5 * np.sum(s[pos] == s[j])) / m
    auc = v10[pos].mean()
    return auc, v10, v01


def delong_paired_test(y, s1, s2):
    """Two-sided DeLong test for paired AUROC difference. Returns auc1, auc2, z, p."""
    auc1, v10_1, v01_1 = fast_delong_components(y, s1)
    auc2, v10_2, v01_2 = fast_delong_components(y, s2)
    pos = np.asarray(y) == 1
    neg = ~pos
    m, k = int(pos.sum()), int(neg.sum())
    v10a, v10b = v10_1[pos], v10_2[pos]
    v01a, v01b = v01_1[neg], v01_2[neg]
    cov10 = np.mean((v10a - v10a.mean()) * (v10b - v10b.mean())) / m
    cov01 = np.mean((v01a - v01a.mean()) * (v01b - v01b.mean())) / k
    var1 = v10a.var() / m + v01a.var() / k
    var2 = v10b.var() / m + v01b.var() / k
    se = np.sqrt(var1 + var2 - 2 * (cov10 + cov01))
    z = (auc1 - auc2) / se if se > 0 else 0.0
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc1, auc2, float(z), float(p)


def bootstrap_auc_ci(y, s, n_iter=2000, seed=42):
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs = []
    while len(aucs) < n_iter:
        idx = rng.randint(0, n, n)
        yy, ss = y[idx], s[idx]
        if yy.sum() == 0 or yy.sum() == n:
            continue
        aucs.append(roc_auc_score(yy, ss))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


class CoOpTextAdapter(torch.nn.Module):
    """Learnable context tokens inside the frozen BiomedCLIP text encoder."""

    def __init__(self, text_tower, n_ctx: int = 16, classnames=("benign", "malignant")):
        super().__init__()
        self.text = text_tower
        self.n_ctx = n_ctx
        width = self.text.config.hidden_size
        self.pad_id = self.text.config.pad_token_id
        self.ctx = torch.nn.Parameter(torch.empty(n_ctx, width).normal_(std=0.02))

    def set_class_tokens(self, tokens):
        self.register_buffer("class_tokens", tokens)

    def forward(self):
        import torch.nn.functional as F
        ids = self.class_tokens  # [2, L]
        B = ids.shape[0]
        embeds = self.text.transformer.embeddings(ids)
        ctx = self.ctx.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([embeds[:, :1], ctx, embeds[:, 1:]], dim=1)
        mask = torch.cat(
            [
                torch.ones(B, 1 + self.n_ctx, device=ids.device),
                (ids[:, 1:] != self.pad_id).long(),
            ],
            dim=1,
        )
        out = self.text.transformer(inputs_embeds=x, attention_mask=mask)
        pooled = self.text.pooler(out, mask)
        return self.text.proj(pooled)


class ContextPromptEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int = 512, n_ctx: int = 16, condition: bool = False):
        super().__init__()
        self.ctx = torch.nn.Parameter(torch.randn(n_ctx, embed_dim) * 0.02)
        self.cond = torch.nn.Linear(embed_dim, embed_dim) if condition else None
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, visual_feature=None):
        base = self.ctx.mean(0)
        if self.cond is not None and visual_feature is not None:
            base = base + self.cond(visual_feature)
        return self.proj(base)


def main():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from cdf_vlm.biomedclip import load_local_biomedclip
    from cdf_vlm.ddsm import DDSMPairedViewDataset
    from cdf_vlm.io import ensure_dir
    from cdf_vlm.multiview_v2 import PromptAttentionGate
    from cdf_vlm.text_prompts import DEFAULT_KEYWORDS, PathologyPromptEncoder
    from cdf_vlm.birads_kg import KGEnhancedPromptEncoder

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-pair-csv", default="data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv")
    ap.add_argument("--data-root", default="data/processed/ddsm")
    ap.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    ap.add_argument("--out", default="outputs/statistics")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = ensure_dir(Path(args.out))
    print(f"device={device}")

    clip_model, preprocess, tokenizer = load_local_biomedclip(args.biomedclip_dir, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()
    logit_scale = float(clip_model.logit_scale.exp().detach())

    test_ds = DDSMPairedViewDataset(args.test_pair_csv, data_root=args.data_root, image_transform=preprocess)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    print(f"test samples: {len(test_ds)}")

    def encode_image(img):
        f = clip_model.encode_image(img)
        return torch.nn.functional.normalize(f.float(), dim=-1)

    def make_gate_model(encoder):
        gate = PromptAttentionGate(embed_dim=512)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.prompt_encoder = encoder
                self.prompt_gate = gate
                self.classifier = nn.Sequential(
                    nn.LayerNorm(512),
                    nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(256, 1),
                )

            def forward_single(self, image, description):
                feat = encode_image(image)
                prompt = self.prompt_encoder.encode_texts(description, device=image.device)
                feat = self.prompt_gate(feat, prompt)
                return self.classifier(feat).squeeze(-1)

        return M().to(device)

    def make_notext_model():
        enc = KGEnhancedPromptEncoder(embed_dim=512, keywords=DEFAULT_KEYWORDS)
        gate = PromptAttentionGate(embed_dim=512)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.prompt_encoder = enc
                self.prompt_gate = gate
                self.classifier = nn.Sequential(
                    nn.LayerNorm(512),
                    nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(256, 1),
                )

            def forward_single(self, image, description=None):
                feat = encode_image(image)
                return self.classifier(feat).squeeze(-1)

        return M().to(device)

    def make_context_model(condition):
        enc = ContextPromptEncoder(embed_dim=512, n_ctx=16, condition=condition)
        gate = PromptAttentionGate(embed_dim=512)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.prompt_encoder = enc
                self.prompt_gate = gate
                self.classifier = nn.Sequential(
                    nn.LayerNorm(512),
                    nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(256, 1),
                )

            def forward_single(self, image, description=None):
                feat = encode_image(image)
                prompt = self.prompt_encoder(feat if condition else None)
                feat = self.prompt_gate(feat, prompt)
                return self.classifier(feat).squeeze(-1)

        return M().to(device)

    def make_coop_cosine_model():
        adapter = CoOpTextAdapter(clip_model.text, n_ctx=16, classnames=("benign", "malignant")).to(device)
        return adapter

    MODELS = {
        "kgtpca": (
            "outputs/encoder_ablation/kg_enhanced/model.pt",
            lambda: make_gate_model(KGEnhancedPromptEncoder(embed_dim=512, keywords=DEFAULT_KEYWORDS)),
            "sigmoid",
        ),
        "baseline": (
            "outputs/encoder_ablation/baseline/model.pt",
            lambda: make_gate_model(PathologyPromptEncoder(embed_dim=512, keywords=DEFAULT_KEYWORDS)),
            "sigmoid",
        ),
        "coop_gate": (
            "outputs/coop_baselines/coop_gate/model.pt",
            lambda: make_context_model(condition=False),
            "sigmoid",
        ),
        "cocoop_gate": (
            "outputs/coop_baselines/cocoop_gate/model.pt",
            lambda: make_context_model(condition=True),
            "sigmoid",
        ),
        "no_text": (
            "outputs/prior_robustness/no_text/model.pt",
            make_notext_model,
            "sigmoid",
        ),
        "seed123_kg": (
            "outputs/prior_robustness/seed123_kg/model.pt",
            lambda: make_gate_model(KGEnhancedPromptEncoder(embed_dim=512, keywords=DEFAULT_KEYWORDS)),
            "sigmoid",
        ),
        "seed123_base": (
            "outputs/prior_robustness/seed123_base/model.pt",
            lambda: make_gate_model(KGEnhancedPromptEncoder(embed_dim=512, keywords=DEFAULT_KEYWORDS)),
            "sigmoid",
        ),
        "seed456_kg": (
            "outputs/prior_robustness/seed456_kg/model.pt",
            lambda: make_gate_model(KGEnhancedPromptEncoder(embed_dim=512, keywords=DEFAULT_KEYWORDS)),
            "sigmoid",
        ),
        "seed456_base": (
            "outputs/prior_robustness/seed456_base/model.pt",
            lambda: make_gate_model(KGEnhancedPromptEncoder(embed_dim=512, keywords=DEFAULT_KEYWORDS)),
            "sigmoid",
        ),
    }

    predictions = {}
    for name, (ckpt, builder, scoring) in MODELS.items():
        ckpt = Path(ckpt)
        if not ckpt.exists():
            print(f"[skip] {name}: {ckpt} missing")
            continue
        model = builder()
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd)
        model.eval()

        if name == "coop_cosine":
            raise NotImplementedError("coop_cosine handled separately")

        probs_all, labels_all = [], []
        with torch.no_grad():
            for batch in test_loader:
                cc = model.forward_single(batch["cc_image"].to(device), list(batch["cc_description"]))
                mlo = model.forward_single(batch["mlo_image"].to(device), list(batch["mlo_description"]))
                probs = torch.sigmoid((cc + mlo) * 0.5).cpu().numpy()
                probs_all.extend(probs.tolist())
                labels_all.extend(batch["label"].cpu().numpy().tolist())
        predictions[name] = {"label": labels_all, "prob": probs_all}
        np.savetxt(out_dir / f"pred_{name}.csv", np.column_stack([labels_all, probs_all]),
                   delimiter=",", header="label,prob", comments="")
        print(f"{name}: collected {len(probs_all)} predictions")

    # CoOp cosine: special scoring path
    ckpt = Path("outputs/coop_baselines/coop_cosine/model.pt")
    if ckpt.exists():
        adapter = make_coop_cosine_model()
        adapter.set_class_tokens(tokenizer(["benign", "malignant"]).to(device))
        adapter.load_state_dict(torch.load(ckpt, map_location=device))
        adapter.eval()

        def class_text_features():
            t = adapter()
            return torch.nn.functional.normalize(t.float(), dim=-1)

        probs_all, labels_all = [], []
        with torch.no_grad():
            for batch in test_loader:
                v_cc = encode_image(batch["cc_image"].to(device))
                v_mlo = encode_image(batch["mlo_image"].to(device))
                v = torch.nn.functional.normalize((v_cc + v_mlo) * 0.5, dim=-1)
                t = class_text_features()
                logits = (v @ t.T) * logit_scale
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs_all.extend(probs.tolist())
                labels_all.extend(batch["label"].cpu().numpy().tolist())
        predictions["coop_cosine"] = {"label": labels_all, "prob": probs_all}
        np.savetxt(out_dir / "pred_coop_cosine.csv", np.column_stack([labels_all, probs_all]),
                   delimiter=",", header="label,prob", comments="")
        print("coop_cosine: collected", len(probs_all))

    # Consistency check: all labels identical
    ref_label = next(iter(predictions.values()))["label"]
    for name, d in predictions.items():
        assert d["label"] == ref_label, f"label mismatch for {name}"
    labels = np.array(ref_label)

    from sklearn.metrics import roc_auc_score
    summary = {}
    for name, d in predictions.items():
        p = np.array(d["prob"])
        auroc = float(roc_auc_score(labels, p))
        lo, hi = bootstrap_auc_ci(labels, p)
        summary[name] = {"auroc": round(auroc, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"{name}: AUROC={auroc:.4f} 95% CI [{lo:.4f}, {hi:.4f}]")

    # Paired DeLong vs kgtpca
    p_kg = np.array(predictions["kgtpca"]["prob"])
    delong = {}
    for name, d in predictions.items():
        if name == "kgtpca":
            continue
        auc1, auc2, z, p = delong_paired_test(labels, p_kg, np.array(d["prob"]))
        delong[name] = {"auc_kgtpca": round(auc1, 4), "auc_other": round(auc2, 4),
                        "z": round(z, 3), "p": float(p)}
        print(f"DeLong kgtpca vs {name}: {auc1:.4f} vs {auc2:.4f}, z={z:.3f}, p={p:.6g}")

    (out_dir / "statistics.json").write_text(json.dumps(
        {"auroc_ci": summary, "delong_vs_kgtpca": delong}, indent=2))
    print("saved:", out_dir / "statistics.json")


if __name__ == "__main__":
    main()
