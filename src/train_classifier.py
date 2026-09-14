"""
Phase 5 / 9: Train (or retrain) the DenseNet-121 classifier under one of the
four conditions (C0/C1/C2/C3), using the two-stage fine-tuning schedule.

Usage:
    python src/train_classifier.py --manifest data/manifests/splits_seed17.csv \
        --seed 17 --condition C0

    python src/train_classifier.py --manifest data/manifests/splits_seed17.csv \
        --seed 17 --condition C3 \
        --synthetic_manifest data/manifests/synthetic_seed17_C3.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(__file__))
from config import NUM_WORKERS  # noqa: E402
from config import (  # noqa: E402
    CLASS_NAMES, CLF_BACKBONE_LR, CLF_BATCH_SIZE, CLF_EARLY_STOP_PATIENCE,
    CLF_HEAD_LR, CLF_LABEL_SMOOTHING, CLF_STAGE_A_EPOCHS, CLF_STAGE_B_MAX_EPOCHS,
    CLF_WARMUP_EPOCHS, CLF_WEIGHT_DECAY, NUM_CLASSES, OUTPUT_CHECKPOINT_DIR,
    OUTPUT_LOG_DIR, OUTPUT_PREDICTION_DIR,
)
from datasets import HAM10000ClassifierDataset  # noqa: E402
from models_densenet import (  # noqa: E402
    build_densenet121, freeze_backbone, get_param_groups, unfreeze_final_block,
)


def make_weighted_sampler(df: pd.DataFrame):
    """Classical control (C1): inverse-frequency weighted sampler, no VAE."""
    class_counts = df["dx"].value_counts()
    weights = df["dx"].map(lambda c: 1.0 / class_counts[c]).values
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), macro_f1


def train_classifier(manifest_path: str, seed: int, condition: str,
                      synthetic_manifest_path: str = None,
                      max_epochs_stage_b: int = CLF_STAGE_B_MAX_EPOCHS,
                      real_only_finetune_epochs: int = 0,
                      device=None):
    """
    real_only_finetune_epochs: for conditions using synthetic data (C2/C3),
        an optional short final phase (recommended: 3-5) training on ONLY
        real images at a reduced learning rate, after the main combined-data
        training. Purpose: the combined phase lets the model benefit from
        the extra synthetic volume, but can pick up subtle synthetic-vs-real
        texture shortcuts (evidenced by a large train/val gap); this final
        real-only phase re-anchors the decision boundary to the real image
        distribution specifically. Ignored (no-op) for C0/C1.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(manifest_path)
    train_df = df[df["partition"] == "train"].copy()
    val_df = df[df["partition"] == "val"].copy()

    synthetic_df = None
    if synthetic_manifest_path:
        synthetic_df = pd.read_csv(synthetic_manifest_path)
        synthetic_df = synthetic_df[synthetic_df["review_status"] != "rejected"]

    train_ds = HAM10000ClassifierDataset(train_df, train=True, synthetic_df=synthetic_df)
    val_ds = HAM10000ClassifierDataset(val_df, train=False)

    if condition == "C1":
        combined_df = train_ds.df  # real only for C1 (no VAE synthetic)
        sampler = make_weighted_sampler(combined_df)
        train_loader = DataLoader(train_ds, batch_size=CLF_BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        train_loader = DataLoader(train_ds, batch_size=CLF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_loader = DataLoader(val_ds, batch_size=CLF_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_densenet121(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=CLF_LABEL_SMOOTHING)

    run_tag = f"{condition}_seed{seed}"
    ckpt_path = os.path.join(OUTPUT_CHECKPOINT_DIR, f"densenet121_{run_tag}.pt")
    log_rows = []

    # ---- Stage A: frozen backbone, head only ----
    model = freeze_backbone(model)
    optimizer = torch.optim.AdamW(get_param_groups(model, CLF_HEAD_LR, CLF_BACKBONE_LR),
                                   weight_decay=CLF_WEIGHT_DECAY)
    for epoch in range(1, CLF_STAGE_A_EPOCHS + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[{run_tag}] StageA epoch {epoch}/{CLF_STAGE_A_EPOCHS} "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")
        log_rows.append({"stage": "A", "epoch": epoch, "train_loss": train_loss,
                          "train_macro_f1": train_f1, "val_loss": val_loss, "val_macro_f1": val_f1})

    # ---- Stage B: unfreeze final block, discriminative LR fine-tuning ----
    model = unfreeze_final_block(model)
    optimizer = torch.optim.AdamW(get_param_groups(model, CLF_HEAD_LR, CLF_BACKBONE_LR),
                                   weight_decay=CLF_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs_stage_b)

    best_val_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs_stage_b + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"[{run_tag}] StageB epoch {epoch}/{max_epochs_stage_b} "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")
        log_rows.append({"stage": "B", "epoch": epoch, "train_loss": train_loss,
                          "train_macro_f1": train_f1, "val_loss": val_loss, "val_macro_f1": val_f1})

        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "condition": condition, "seed": seed, "val_macro_f1": val_f1}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CLF_EARLY_STOP_PATIENCE:
                print(f"[{run_tag}] early stopping at epoch {epoch} (best val_f1={best_val_f1:.4f})")
                break

    log_df = pd.DataFrame(log_rows)
    log_csv = os.path.join(OUTPUT_LOG_DIR, f"densenet121_{run_tag}_log.csv")
    log_df.to_csv(log_csv, index=False)

    # ---- Stage C (optional): short real-only fine-tune to correct any
    # synthetic-vs-real shortcut the model picked up during combined training ----
    if real_only_finetune_epochs > 0 and condition in ("C2", "C3"):
        print(f"[{run_tag}] Starting Stage C: real-only fine-tune "
              f"({real_only_finetune_epochs} epochs, reduced LR)")

        # Reload best combined-training checkpoint as the starting point.
        best_state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best_state["model_state"])

        real_only_ds = HAM10000ClassifierDataset(train_df, train=True, synthetic_df=None)
        real_only_loader = DataLoader(real_only_ds, batch_size=CLF_BATCH_SIZE, shuffle=True,
                                       num_workers=NUM_WORKERS)

        stage_c_optimizer = torch.optim.AdamW(
            get_param_groups(model, CLF_HEAD_LR * 0.1, CLF_BACKBONE_LR * 0.1),
            weight_decay=CLF_WEIGHT_DECAY,
        )

        for epoch in range(1, real_only_finetune_epochs + 1):
            train_loss, train_f1 = run_epoch(model, real_only_loader, criterion, stage_c_optimizer, device, train=True)
            val_loss, val_f1 = run_epoch(model, val_loader, criterion, stage_c_optimizer, device, train=False)

            print(f"[{run_tag}] StageC epoch {epoch}/{real_only_finetune_epochs} "
                  f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")
            log_rows.append({"stage": "C", "epoch": epoch, "train_loss": train_loss,
                              "train_macro_f1": train_f1, "val_loss": val_loss, "val_macro_f1": val_f1})

            if val_f1 > best_val_f1 + 1e-5:
                best_val_f1 = val_f1
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "condition": condition, "seed": seed, "val_macro_f1": val_f1,
                            "stage": "C"}, ckpt_path)
                print(f"[{run_tag}] StageC improved val_f1 to {best_val_f1:.4f} -- checkpoint updated.")

        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(log_csv, index=False)

    print(f"[{run_tag}] done. best_val_macro_f1={best_val_f1:.4f}. Checkpoint: {ckpt_path}")
    return ckpt_path, log_csv


def main():
    parser = argparse.ArgumentParser(description="Train/retrain DenseNet-121 under one condition.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=["C0", "C1", "C2", "C3"], required=True)
    parser.add_argument("--synthetic_manifest", default=None,
                         help="Required for C2/C3, path to generate_synthetic.py output CSV.")
    parser.add_argument("--max_epochs_stage_b", type=int, default=CLF_STAGE_B_MAX_EPOCHS)
    parser.add_argument("--real_only_finetune_epochs", type=int, default=0,
                         help="Optional short real-only polish phase after combined training "
                              "(C2/C3 only). Recommended: 3-5.")
    args = parser.parse_args()

    if args.condition in ("C2", "C3") and not args.synthetic_manifest:
        raise ValueError(f"Condition {args.condition} requires --synthetic_manifest "
                          f"(output of generate_synthetic.py).")

    train_classifier(args.manifest, args.seed, args.condition,
                      synthetic_manifest_path=args.synthetic_manifest,
                      max_epochs_stage_b=args.max_epochs_stage_b,
                      real_only_finetune_epochs=args.real_only_finetune_epochs)


if __name__ == "__main__":
    main()
