from __future__ import annotations

import argparse
import sys

import numpy as np


def _single_forward(model, batch, device):
    return model(
        image=batch["image"].to(device),
        mask=batch["mask"].to(device),
        description=list(batch["description"]),
    )


def _pair_forward(model, batch, device):
    return model(
        cc_image=batch["cc_image"].to(device),
        cc_mask=batch["cc_mask"].to(device),
        cc_description=list(batch["cc_description"]),
        mlo_image=batch["mlo_image"].to(device),
        mlo_mask=batch["mlo_mask"].to(device),
        mlo_description=list(batch["mlo_description"]),
        side_ids=list(batch["side_id"]),
    )


def _label_smoothing(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Smooth binary targets from {0,1} toward 0.5."""
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint DDSM training V2: improved aligner + bilateral fusion.")
    parser.add_argument("--single-csv", required=True)
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--val-pair-csv")
    parser.add_argument("--test-pair-csv")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", choices=["biomedclip", "mammoclip"], default="biomedclip")
    parser.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--mammoclip-checkpoint", default="external/Mammo-CLIP/Pre-trained-checkpoints/b2-model-best-epoch-10.tar")
    parser.add_argument("--mammoclip-source-dir", default="external/Mammo-CLIP-src/src/codebase")
    parser.add_argument("--mammoclip-input-size", type=int, nargs=2, default=[760, 456],
                        help="Input resolution H W for Mammo-CLIP (default: 760 456)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-single", type=float, default=1.0)
    parser.add_argument("--lambda-pair", type=float, default=1.0)
    parser.add_argument("--lambda-proj", type=float, default=0.1,
                        help="Weight for cross-view alignment loss (V2 aligner, higher than V1 STN)")
    parser.add_argument("--aligner-warmup-epochs", type=int, default=5,
                        help="Gradually increase alignment loss weight from 0 over this many epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs with no val AUROC improvement)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--vision-rank", type=int, default=32,
                        help="LoRA vision rank (default 32, lower than V1 to reduce overfitting)")
    parser.add_argument("--text-rank", type=int, default=8,
                        help="LoRA text rank (default 8, lower than V1 to reduce overfitting)")
    parser.add_argument("--disable-aligner", action="store_true",
                        help="Disable CrossViewFeatureAligner (V2 replacement for STN)")
    parser.add_argument("--disable-bilateral", action="store_true",
                        help="Disable ViewAwareBilateralFusion (V2 replacement for GNN)")
    parser.add_argument("--disable-text-prompts", action="store_true")
    parser.add_argument("--use-consistency", action="store_true",
                        help="Enable cross-view consistency loss (MSE between CC and MLO predictions)")
    parser.add_argument("--lambda-consistency", type=float, default=0.05,
                        help="Weight for cross-view consistency loss (default 0.05)")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="MixUp alpha for feature-level mixing (0=disabled, 0.2 recommended for small datasets)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0=disabled, 0.05-0.1 recommended to improve ECE)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default 1.0, helps LoRA stability)")
    parser.add_argument("--pos-weight", type=float, default=None, help="Positive class weight for BCEWithLogitsLoss (e.g., neg_count/pos_count)")
    parser.add_argument("--lora-dropout", type=float, default=0.25, help="Dropout rate for LoRA adapters (increased default 0.25)")
    parser.add_argument("--device")
    parser.add_argument("--mock-backbone", action="store_true", help="Use a tiny local backbone for smoke tests.")
    args = parser.parse_args()

    import gc
    import itertools
    import os

    import pandas as pd
    import psutil
    import torch
    from tqdm import tqdm
    from torch import nn
    from torch.utils.data import DataLoader

    from cdf_vlm.biomedclip import load_local_biomedclip, load_mock_biomedclip
    from cdf_vlm.config import save_json
    from cdf_vlm.ddsm import DDSMPairedViewDataset, DDSMSingleViewDataset
    from cdf_vlm.io import ensure_dir, seed_everything
    from cdf_vlm.lora import inject_decoupled_lora
    from cdf_vlm.metrics import binary_metrics
    from cdf_vlm.multiview_v2 import DDSMGeometryPromptModelV2, DDSMSingleViewPromptModelV2

    seed_everything(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")
    out_dir = ensure_dir(args.output_dir)

    if args.backbone == "mammoclip":
        from cdf_vlm.mammoclip_backbone import MammoCLIPWrapper, build_mammoclip_preprocess

        if args.use_lora:
            print("[WARN] --use-lora is not supported for Mammo-CLIP (CNN backbone); ignoring.")
        # Bilateral fusion empirically hurts Mammo-CLIP (CNN features), warn if enabled
        if not args.disable_bilateral:
            print("[WARN] Bilateral fusion shows negative effect on Mammo-CLIP (ΔAUROC≈-0.003).")
            print("[WARN] Consider adding --disable-bilateral for optimal Mammo-CLIP performance.")
        input_size = tuple(args.mammoclip_input_size)
        print(f"Loading Mammo-CLIP checkpoint ({args.mammoclip_checkpoint})...")
        sys.stdout.flush()
        clip_model = MammoCLIPWrapper(
            checkpoint=args.mammoclip_checkpoint,
            source_dir=args.mammoclip_source_dir,
            device=device,
            input_size=input_size,
        )
        preprocess = build_mammoclip_preprocess(image_size=input_size)
        print(f"Using Mammo-CLIP backbone (EfficientNet-B2, input={input_size[0]}x{input_size[1]}, 512-dim)")
        sys.stdout.flush()
    elif args.mock_backbone:
        clip_model, preprocess, _ = load_mock_biomedclip(device=device)
    else:
        print(f"Loading BiomedCLIP from {args.biomedclip_dir}...")
        sys.stdout.flush()
        clip_model, preprocess, _ = load_local_biomedclip(args.biomedclip_dir, device=device)
        if args.use_lora:
            report = inject_decoupled_lora(
                clip_model,
                vision_rank=args.vision_rank,
                text_rank=args.text_rank,
                dropout=args.lora_dropout,
            )
            print(f"Injected LoRA: vision={report.vision_modules}, text={report.text_modules}")

    print("Loading datasets...")
    sys.stdout.flush()
    single_dataset = DDSMSingleViewDataset(args.single_csv, data_root=args.data_root, image_transform=preprocess)
    pair_dataset = DDSMPairedViewDataset(args.pair_csv, data_root=args.data_root, image_transform=preprocess)
    print(f"Single-view samples: {len(single_dataset)}, paired samples: {len(pair_dataset)}")
    sys.stdout.flush()
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

    single_model = DDSMSingleViewPromptModelV2(
        clip_model=clip_model,
        freeze_backbone=not args.use_lora,
        use_text_prompts=not args.disable_text_prompts,
    ).to(device)
    pair_model = DDSMGeometryPromptModelV2(
        clip_model=clip_model,
        freeze_backbone=not args.use_lora,
        use_aligner=not args.disable_aligner,
        use_bilateral=not args.disable_bilateral,
        use_text_prompts=not args.disable_text_prompts,
        use_consistency=args.use_consistency,
    ).to(device)
    pos_weight = torch.tensor([args.pos_weight], device=device) if args.pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    params = []
    seen = set()
    for param in itertools.chain(single_model.parameters(), pair_model.parameters()):
        if param.requires_grad and id(param) not in seen:
            params.append(param)
            seen.add(id(param))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    @torch.no_grad()
    def evaluate(eval_loader, split_name: str):
        pair_model.eval()
        labels_all = []
        probs_all = []
        rows = []
        losses = []
        for batch in tqdm(eval_loader, desc=f"Eval {split_name}", leave=False):
            labels = batch["label"].to(device)
            out = _pair_forward(pair_model, batch, device)
            cls_loss = criterion(out["logits"], labels).mean()
            loss = (
                cls_loss
                + args.lambda_proj * out["projection_loss"]
                + args.lambda_consistency * out.get("consistency_loss", 0.0)
            )
            losses.append(float(loss.detach().cpu()))
            probs = torch.sigmoid(out["logits"])
            labels_np = labels.detach().cpu().numpy().tolist()
            probs_np = probs.detach().cpu().numpy().tolist()
            labels_all.extend(labels_np)
            probs_all.extend(probs_np)
            for i, side_id in enumerate(list(batch["side_id"])):
                rows.append(
                    {
                        "split": split_name,
                        "side_id": side_id,
                        "label": labels_np[i],
                        "prob": probs_np[i],
                        "cc_description": list(batch["cc_description"])[i],
                        "mlo_description": list(batch["mlo_description"])[i],
                    }
                )
        metrics = binary_metrics(labels_all, probs_all)
        metrics["loss"] = sum(losses) / max(len(losses), 1)
        return metrics, rows

    def _mem_info():
        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / (1024**3)
        gpu_gb = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"RAM={rss_gb:.1f}G GPU={gpu_gb:.1f}G(reserved={gpu_reserved:.1f}G)"

    history = []
    best_metric = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    print(f"Starting training, initial memory: {_mem_info()}")
    sys.stdout.flush()
    for epoch in range(1, args.epochs + 1):
        # Aligner warmup: gradually increase alignment loss weight
        if epoch <= args.aligner_warmup_epochs:
            warmup_factor = epoch / max(args.aligner_warmup_epochs, 1)
        else:
            warmup_factor = 1.0
        effective_lambda_proj = args.lambda_proj * warmup_factor

        single_model.train()
        pair_model.train()
        losses = []
        single_iter = iter(single_loader)
        pair_iter = iter(pair_loader)
        steps = max(len(pair_loader), len(single_loader))
        pbar = tqdm(range(steps), desc=f"Epoch {epoch:2d}/{args.epochs}", unit="step")
        for step_idx in pbar:
            try:
                single_batch = next(single_iter)
            except StopIteration:
                single_iter = iter(single_loader)
                single_batch = next(single_iter)
            try:
                pair_batch = next(pair_iter)
            except StopIteration:
                pair_iter = iter(pair_loader)
                pair_batch = next(pair_iter)

            # --- MixUp on paired data (image-level, symmetric across views) ---
            pair_labels = pair_batch["label"].to(device)
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                idx = torch.randperm(pair_labels.size(0))  # CPU indices for CPU batch tensors
                # Mix images and masks
                for key in ("cc_image", "cc_mask", "mlo_image", "mlo_mask"):
                    orig = pair_batch[key]
                    pair_batch[key] = lam * orig + (1.0 - lam) * orig[idx]
                pair_labels = lam * pair_labels + (1.0 - lam) * pair_labels[idx]

            single_labels = single_batch["label"].to(device)
            single_out = _single_forward(single_model, single_batch, device)
            pair_out = _pair_forward(pair_model, pair_batch, device)

            # Label smoothing
            single_targets = _label_smoothing(single_labels, args.label_smoothing)
            pair_targets = _label_smoothing(pair_labels, args.label_smoothing)

            single_loss = criterion(single_out["logits"], single_targets).mean()
            pair_cls_loss = criterion(pair_out["logits"], pair_targets).mean()
            loss = (
                args.lambda_single * single_loss
                + args.lambda_pair * pair_cls_loss
                + effective_lambda_proj * pair_out["projection_loss"]
                + args.lambda_consistency * pair_out.get("consistency_loss", 0.0)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            del single_out, pair_out, single_loss, pair_cls_loss, loss
            del single_batch, pair_batch
            if step_idx % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if step_idx % 50 == 0:
                pbar.set_postfix(loss=f"{losses[-1]:.4f}", mem=_mem_info())

        row = {"epoch": epoch, "loss": sum(losses) / max(len(losses), 1), "lambda_proj": effective_lambda_proj}
        if val_loader is not None:
            val_metrics, _ = evaluate(val_loader, "val")
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = val_metrics.get("auroc", val_metrics.get("auprc", 0.0))
        else:
            score = -row["loss"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "single_model": {k: v.detach().cpu().clone() for k, v in single_model.state_dict().items()},
                "pair_model": {k: v.detach().cpu().clone() for k, v in pair_model.state_dict().items()},
            }
        else:
            patience_counter += 1
        history.append(row)
        print(row)
        sys.stdout.flush()

        # Early stopping
        if patience_counter >= args.patience and epoch > args.aligner_warmup_epochs:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    if best_state is not None:
        single_model.load_state_dict(best_state["single_model"])
        pair_model.load_state_dict(best_state["pair_model"])

    predictions = []
    metrics = {"best_epoch": best_epoch, "best_metric": best_metric}
    train_eval_loader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_metrics, train_rows = evaluate(train_eval_loader, "train")
    metrics["train"] = train_metrics
    predictions.extend(train_rows)
    if val_loader is not None:
        val_metrics, val_rows = evaluate(val_loader, "val")
        metrics["val"] = val_metrics
        predictions.extend(val_rows)
    if test_loader is not None:
        test_metrics, test_rows = evaluate(test_loader, "test")
        metrics["test"] = test_metrics
        predictions.extend(test_rows)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame(predictions).to_csv(out_dir / "predictions.csv", index=False)
    save_json(metrics, out_dir / "metrics.json")
    torch.save(
        {
            "single_model_state": single_model.state_dict(),
            "pair_model_state": pair_model.state_dict(),
            "args": vars(args),
        },
        out_dir / "model.pt",
    )
    print(f"Wrote joint training outputs to {out_dir}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
