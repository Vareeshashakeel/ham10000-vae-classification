"""
Phase 10 / 11: Evaluate trained checkpoints on the frozen real-only test
set, compute the full metric suite, and (optionally) a paired bootstrap
comparison between two conditions (e.g. C0 vs C3).

Usage:
    python src/evaluate.py --manifest data/manifests/splits_seed17.csv \
        --checkpoint outputs/checkpoints/densenet121_C0_seed17.pt \
        --condition C0 --seed 17

    python src/evaluate.py --compare \
        --pred_a outputs/predictions/preds_C0_seed17.csv \
        --pred_b outputs/predictions/preds_C3_seed17.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, NUM_CLASSES, NUM_WORKERS, OUTPUT_FIGURE_DIR, OUTPUT_PREDICTION_DIR  # noqa: E402
from datasets import HAM10000ClassifierDataset  # noqa: E402
from models_densenet import build_densenet121  # noqa: E402


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return ece


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray):
    accuracy = (labels == preds).mean()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    try:
        macro_auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        macro_auroc = float("nan")

    auprc_per_class = []
    for c in range(NUM_CLASSES):
        y_true_c = (labels == c).astype(int)
        try:
            auprc_per_class.append(average_precision_score(y_true_c, probs[:, c]))
        except ValueError:
            auprc_per_class.append(float("nan"))

    ece = expected_calibration_error(probs, labels)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    per_class_df = pd.DataFrame({
        "class": CLASS_NAMES, "precision": precision, "recall": recall,
        "f1": f1_per_class, "support": support, "auprc": auprc_per_class,
    })

    summary = {
        "accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc, "macro_auroc": macro_auroc, "ece": ece,
    }
    return summary, per_class_df, cm


def predict_on_test(checkpoint_path: str, manifest_path: str, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(manifest_path)
    test_df = df[df["partition"] == "test"].copy()

    ds = HAM10000ClassifierDataset(test_df, train=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

    model = build_densenet121(num_classes=NUM_CLASSES, pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)

    # Attach group id for paired bootstrap later.
    group_ids = test_df["lesion_id"].values if "lesion_id" in test_df.columns else np.arange(len(labels))

    return probs, preds, labels, group_ids


def paired_bootstrap_ci(labels, preds_a, preds_b, group_ids, n_boot=2000, seed=0, metric="macro_f1"):
    """95% CI for macro-F1 difference (B - A), resampling by lesion group."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(group_ids)
    diffs = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask_idx = np.concatenate([np.where(group_ids == g)[0] for g in sampled_groups])
        f1_a = f1_score(labels[mask_idx], preds_a[mask_idx], average="macro", zero_division=0)
        f1_b = f1_score(labels[mask_idx], preds_b[mask_idx], average="macro", zero_division=0)
        diffs.append(f1_b - f1_a)
    diffs = np.array(diffs)
    return {
        "mean_diff": diffs.mean(),
        "ci_lower": np.percentile(diffs, 2.5),
        "ci_upper": np.percentile(diffs, 97.5),
    }


def plot_confusion_matrix(cm, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a classifier checkpoint or compare two conditions.")
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--pred_a")
    parser.add_argument("--pred_b")
    args = parser.parse_args()

    if args.compare:
        df_a = pd.read_csv(args.pred_a)
        df_b = pd.read_csv(args.pred_b)
        labels = df_a["label"].values
        result = paired_bootstrap_ci(
            labels, df_a["pred"].values, df_b["pred"].values, df_a["group_id"].values
        )
        print("Paired bootstrap macro-F1 difference (B - A):")
        print(result)
        return

    probs, preds, labels, group_ids = predict_on_test(args.checkpoint, args.manifest)
    summary, per_class_df, cm = compute_metrics(labels, preds, probs)

    run_tag = f"{args.condition}_seed{args.seed}"
    print(f"=== Results: {run_tag} ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    print(per_class_df.to_string(index=False))

    pred_df = pd.DataFrame({
        "label": labels, "pred": preds, "group_id": group_ids,
    })
    for c in range(NUM_CLASSES):
        pred_df[f"prob_{CLASS_NAMES[c]}"] = probs[:, c]
    pred_csv = os.path.join(OUTPUT_PREDICTION_DIR, f"preds_{run_tag}.csv")
    pred_df.to_csv(pred_csv, index=False)

    per_class_csv = os.path.join(OUTPUT_PREDICTION_DIR, f"per_class_metrics_{run_tag}.csv")
    per_class_df.to_csv(per_class_csv, index=False)

    summary_csv = os.path.join(OUTPUT_PREDICTION_DIR, f"summary_metrics_{run_tag}.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    cm_path = os.path.join(OUTPUT_FIGURE_DIR, f"confusion_matrix_{run_tag}.png")
    plot_confusion_matrix(cm, cm_path, f"Confusion matrix: {run_tag}")

    print(f"Saved predictions -> {pred_csv}")
    print(f"Saved per-class metrics -> {per_class_csv}")
    print(f"Saved summary metrics -> {summary_csv}")
    print(f"Saved confusion matrix plot -> {cm_path}")


if __name__ == "__main__":
    main()
