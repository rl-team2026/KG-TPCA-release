#!/usr/bin/env python3
"""CoOp-family baselines for the AAMAS rebuttal (Reviewer 1, comparison concern).

Three prompt-adaptation baselines on the frozen BiomedCLIP backbone, trained on
CBIS-DDSM under the same data split, optimizer, and early-stopping protocol as
the paper's Table 3/4 runs:

  coop_cosine - classic CoOp: learnable context tokens inserted after [CLS] in
                the class prompt ("benign"/"malignant"), processed by the frozen
                BiomedCLIP text encoder; classification by cosine similarity
                between dual-view averaged visual features and class text
                features. Context tokens are the only trainable parameters.
  coop_gate   - input-independent learned context prompt injected through the
                same FiLM gate pipeline as KG-TPCA (global + local features,
                masks, dual view), but with no BI-RADS keywords and no KG loss.
  cocoop_gate - like coop_gate but the context prompt is conditioned on the
                current view's visual feature (CoCoOp-style).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


class CoOpTextAdapter(torch.nn.Module):
    """Learnable context tokens inside the frozen BiomedCLIP text encoder."""

    def __init__(self, text_tower, n_ctx: int = 16, classnames=("benign", "malignant")):
        super().__init__()
        import torch
        self.text = text_tower
        self.n_ctx = n_ctx
        width = self.text.config.hidden_size
        self.pad_id = self.text.config.pad_token_id
        self.ctx = torch.nn.Parameter(torch.empty(n_ctx, width).normal_(std=0.02))

    def set_class_tokens(self, tokens):
        self.register_buffer("class_tokens", tokens)

    def forward(self):
        import torch
        import torch.nn.functional as F
        ids = self.class_tokens  # [2, L]
        B = ids.shape[0]
        embeds = self.text.transformer.embeddings(ids)  # [B, L, D]
        ctx = self.ctx.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]
        x = torch.cat([embeds[:, :1], ctx, embeds[:, 1:]], dim=1)  # insert after [CLS]
        mask = torch.cat(
            [
                torch.ones(B, 1 + self.n_ctx, device=ids.device),
                (ids[:, 1:] != self.pad_id).long(),
            ],
            dim=1,
        )
        out = self.text.transformer(inputs_embeds=x, attention_mask=mask)
        pooled = self.text.pooler(out, mask)
        return self.text.proj(pooled)  # [2, D]


class ContextPromptEncoder(torch.nn.Module):
    """Input-independent (CoOp-style) or visual-conditioned (CoCoOp-style) context prompt."""

    def __init__(self, embed_dim: int = 512, n_ctx: int = 16, condition: bool = False):
        super().__init__()
        import torch
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


def run_coop_cosine(args, out_dir: Path) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader

    from cdf_vlm.biomedclip import load_local_biomedclip
    from cdf_vlm.config import save_json
    from cdf_vlm.ddsm import DDSMPairedViewDataset, DDSMSingleViewDataset
    from cdf_vlm.io import ensure_dir, seed_everything
    from cdf_vlm.metrics import binary_metrics

    seed_everything(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = ensure_dir(out_dir)

    clip_model, preprocess, tokenizer = load_local_biomedclip(args.biomedclip_dir, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    adapter = CoOpTextAdapter(clip_model.text, n_ctx=args.n_ctx, classnames=("benign", "malignant")).to(device)
    class_tokens = tokenizer(["benign", "malignant"]).to(device)
    adapter.set_class_tokens(class_tokens)

    logit_scale = float(clip_model.logit_scale.exp().detach())

    def encode_image(img):
        f = clip_model.encode_image(img)
        return F.normalize(f.float(), dim=-1)

    def class_text_features():
        t = adapter()
        return F.normalize(t.float(), dim=-1)  # [2, D]

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

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    def logits_and_loss(v, labels):
        t = class_text_features()  # [2, D]
        logits = (v @ t.T) * logit_scale  # [B, 2]
        loss = criterion(logits, labels.long())
        return logits, loss

    def eval_probs(loader):
        adapter.eval()
        probs_all, labels_all = [], []
        with torch.no_grad():
            for batch in loader:
                v_cc = encode_image(batch["cc_image"].to(device))
                v_mlo = encode_image(batch["mlo_image"].to(device))
                v = F.normalize((v_cc + v_mlo) * 0.5, dim=-1)
                t = class_text_features()
                logits = (v @ t.T) * logit_scale
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs_all.extend(probs.tolist())
                labels_all.extend(batch["label"].cpu().numpy().tolist())
        return labels_all, probs_all

    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
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

            v_s = encode_image(s_batch["image"].to(device))
            _, s_loss = logits_and_loss(v_s, s_batch["label"].to(device))

            v_cc = encode_image(p_batch["cc_image"].to(device))
            v_mlo = encode_image(p_batch["mlo_image"].to(device))
            v_p = F.normalize((v_cc + v_mlo) * 0.5, dim=-1)
            _, p_loss = logits_and_loss(v_p, p_batch["label"].to(device))

            loss = s_loss + p_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if val_loader is not None:
            labels_all, probs_all = eval_probs(val_loader)
            val_metrics = binary_metrics(labels_all, probs_all)
            val_auroc = val_metrics["auroc"]
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"Epoch {epoch:2d}: loss={np.mean(losses):.4f} val_auroc={val_auroc:.4f} best={best_auroc:.4f}")
            if patience_counter >= args.patience and epoch > 3:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            print(f"Epoch {epoch:2d}: loss={np.mean(losses):.4f}")

    if best_state is not None:
        adapter.load_state_dict(best_state)

    test_metrics = {}
    if test_loader is not None:
        labels_all, probs_all = eval_probs(test_loader)
        test_metrics = binary_metrics(labels_all, probs_all)
    test_metrics["trainable_params"] = sum(p.numel() for p in adapter.parameters())
    save_json(test_metrics, out_dir / "metrics.json")
    torch.save(adapter.state_dict(), out_dir / "model.pt")
    print(f"coop_cosine test: AUROC={test_metrics.get('auroc',0):.4f} "
          f"AUPRC={test_metrics.get('auprc',0):.4f} F1={test_metrics.get('f1',0):.4f} "
          f"ECE={test_metrics.get('ece',0):.4f}")
    return test_metrics


def run_gate_variant(args, out_dir: Path, condition: bool) -> dict:
    import gc
    import numpy as np
    import torch
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

    clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    embed_dim = 512
    prompt_encoder = ContextPromptEncoder(embed_dim=embed_dim, n_ctx=args.n_ctx, condition=condition)
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

        def forward_single(self, image, description=None):
            feat = self.encode_image(image)
            prompt = self.prompt_encoder(feat if condition else None)
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

            s_logits = model.forward_single(s_batch["image"].to(device))
            s_loss = criterion(s_logits, s_batch["label"].to(device))
            cc_logits = model.forward_single(p_batch["cc_image"].to(device))
            mlo_logits = model.forward_single(p_batch["mlo_image"].to(device))
            p_loss = criterion((cc_logits + mlo_logits) * 0.5, p_batch["label"].to(device))

            loss = s_loss + p_loss
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
                    cc = model.forward_single(batch["cc_image"].to(device))
                    mlo = model.forward_single(batch["mlo_image"].to(device))
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
                cc = model.forward_single(batch["cc_image"].to(device))
                mlo = model.forward_single(batch["mlo_image"].to(device))
                probs = torch.sigmoid((cc + mlo) * 0.5).cpu().numpy()
                probs_all.extend(probs.tolist())
                labels_all.extend(batch["label"].cpu().numpy().tolist())
        test_metrics = binary_metrics(labels_all, probs_all)
    test_metrics["trainable_params"] = n_trainable
    save_json(test_metrics, out_dir / "metrics.json")
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"{'cocoop' if condition else 'coop'}_gate test: AUROC={test_metrics.get('auroc',0):.4f} "
          f"AUPRC={test_metrics.get('auprc',0):.4f} F1={test_metrics.get('f1',0):.4f} "
          f"ECE={test_metrics.get('ece',0):.4f}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="CoOp-family baselines on frozen BiomedCLIP.")
    parser.add_argument("--single-csv", default="data/processed/ddsm/breast_cancer_train.csv")
    parser.add_argument("--pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_train.csv")
    parser.add_argument("--val-pair-csv", default="data/processed/ddsm/breast_cancer_train_cc_mlo_pair_val.csv")
    parser.add_argument("--test-pair-csv", default="data/processed/ddsm/breast_cancer_test_cc_mlo_pair.csv")
    parser.add_argument("--data-root", default="data/processed/ddsm")
    parser.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--output-dir", default="outputs/coop_baselines")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--runs", nargs="+", default=["coop_cosine", "coop_gate", "cocoop_gate"],
                        choices=["coop_cosine", "coop_gate", "cocoop_gate"])
    args = parser.parse_args()

    results = {}
    for run_name in args.runs:
        out_dir = Path(args.output_dir) / run_name
        if (out_dir / "metrics.json").exists():
            print(f"skip {run_name} (metrics.json exists)")
            continue
        try:
            if run_name == "coop_cosine":
                results[run_name] = run_coop_cosine(args, out_dir)
            else:
                results[run_name] = run_gate_variant(args, out_dir, condition=(run_name == "cocoop_gate"))
        except Exception as exc:
            print(f"[ERROR] {run_name} failed: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("COOP BASELINE RESULTS")
    print("=" * 70)
    for name, m in results.items():
        print(f"{name:<14} AUROC={m.get('auroc',0):.4f} AUPRC={m.get('auprc',0):.4f} "
              f"F1={m.get('f1',0):.4f} ECE={m.get('ece',0):.4f}")


if __name__ == "__main__":
    main()
