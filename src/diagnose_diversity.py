"""
Diagnostic: quantifies whether VAE-generated synthetic images are
near-duplicates of each other (low diversity) compared to real images of
the same class. This directly tests the "classifier is memorizing a
narrow synthetic pattern rather than learning generalizable features"
hypothesis raised when train accuracy is high but val accuracy lags.

Method: resize images to a small grayscale thumbnail, flatten to a vector,
L2-normalize, and compute mean pairwise cosine similarity within a class's
real images, within its synthetic images, and across real-vs-synthetic.
Higher similarity = less diversity (more redundant / duplicate-like).
This is intentionally simple and dependency-light (no extra packages
beyond what's already used) -- it's a diagnostic signal, not a
publication-grade perceptual metric.

Usage:
    python src/diagnose_diversity.py --manifest data/manifests/splits_seed17.csv \
        --synthetic_manifest data/manifests/synthetic_seed17_C3.csv --seed 17
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, OUTPUT_FIGURE_DIR, OUTPUT_PREDICTION_DIR  # noqa: E402

THUMB_SIZE = 32  # small on purpose -- we want coarse structural similarity, not pixel-exact


def load_feature_vectors(file_paths, max_n=40, seed=0):
    rng = np.random.RandomState(seed)
    paths = list(file_paths)
    if len(paths) > max_n:
        paths = list(rng.choice(paths, size=max_n, replace=False))

    vecs = []
    for p in paths:
        try:
            img = Image.open(p).convert("L").resize((THUMB_SIZE, THUMB_SIZE))
            arr = np.asarray(img, dtype=np.float32).flatten()
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            vecs.append(arr)
        except Exception as e:
            print(f"  (skipped unreadable file {p}: {e})")
    if not vecs:
        return np.zeros((0, THUMB_SIZE * THUMB_SIZE))
    return np.stack(vecs)


def mean_pairwise_cosine_sim(vecs_a, vecs_b=None):
    """If vecs_b is None, computes within-set similarity (excluding self-pairs)."""
    if len(vecs_a) < 2 and vecs_b is None:
        return float("nan")
    if vecs_b is None:
        sim_matrix = vecs_a @ vecs_a.T
        n = len(vecs_a)
        mask = ~np.eye(n, dtype=bool)
        return sim_matrix[mask].mean()
    if len(vecs_a) == 0 or len(vecs_b) == 0:
        return float("nan")
    sim_matrix = vecs_a @ vecs_b.T
    return sim_matrix.mean()


def run_diversity_diagnostic(manifest_path: str, synthetic_manifest_path: str, seed: int):
    df = pd.read_csv(manifest_path)
    train_df = df[df["partition"] == "train"]
    synth_df = pd.read_csv(synthetic_manifest_path)
    synth_df = synth_df[synth_df["review_status"] != "rejected"]

    rows = []
    for cls in CLASS_NAMES:
        real_paths = train_df[train_df["dx"] == cls]["file_path"].tolist()
        synth_paths = synth_df[synth_df["dx"] == cls]["file_path"].tolist()

        real_vecs = load_feature_vectors(real_paths, seed=seed)
        synth_vecs = load_feature_vectors(synth_paths, seed=seed)

        real_internal_sim = mean_pairwise_cosine_sim(real_vecs)
        synth_internal_sim = mean_pairwise_cosine_sim(synth_vecs) if len(synth_vecs) >= 2 else float("nan")
        cross_sim = mean_pairwise_cosine_sim(real_vecs, synth_vecs) if len(synth_vecs) > 0 else float("nan")

        rows.append({
            "class": cls,
            "n_real": len(real_vecs),
            "n_synthetic": len(synth_vecs),
            "real_internal_similarity": real_internal_sim,
            "synthetic_internal_similarity": synth_internal_sim,
            "real_vs_synthetic_similarity": cross_sim,
            "synthetic_more_redundant_than_real": (
                (synth_internal_sim - real_internal_sim) if not np.isnan(synth_internal_sim) else np.nan
            ),
        })

    result_df = pd.DataFrame(rows)
    print("=" * 100)
    print("SYNTHETIC IMAGE DIVERSITY DIAGNOSTIC")
    print("Higher 'similarity' = LESS diverse (images more redundant/near-duplicate-like).")
    print("=" * 100)
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 100)

    flagged = result_df[result_df["synthetic_more_redundant_than_real"] > 0.05]
    if len(flagged) > 0:
        print("Classes where synthetic images are NOTABLY less diverse than real images "
              "(gap > 0.05 -- a plausible contributor to classifier overfitting on these classes):")
        print(flagged[["class", "synthetic_more_redundant_than_real"]].to_string(index=False))
    else:
        print("No class shows a large diversity gap by this metric -- low synthetic diversity "
              "is likely NOT the main explanation here; consider other causes (e.g. sheer volume "
              "of near-identical *counts* even if not near-identical *images*, or distribution "
              "shift between VAE output style and real images).")

    csv_path = os.path.join(OUTPUT_PREDICTION_DIR, f"diversity_diagnostic_seed{seed}.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"\nSaved full table to: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_df = result_df.dropna(subset=["synthetic_internal_similarity"])
        if len(plot_df) > 0:
            x = np.arange(len(plot_df))
            width = 0.35
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(x - width / 2, plot_df["real_internal_similarity"], width, label="Real images (within-class)")
            ax.bar(x + width / 2, plot_df["synthetic_internal_similarity"], width, label="Synthetic images (within-class)")
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df["class"])
            ax.set_ylabel("Mean pairwise cosine similarity\n(higher = less diverse)")
            ax.set_title(f"Real vs. synthetic image diversity per class (seed {seed})")
            ax.legend()
            plt.tight_layout()
            fig_path = os.path.join(OUTPUT_FIGURE_DIR, f"diversity_comparison_seed{seed}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"Saved comparison plot to: {fig_path}")
    except ImportError:
        pass

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Diagnose synthetic vs real image diversity per class.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--synthetic_manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_diversity_diagnostic(args.manifest, args.synthetic_manifest, args.seed)


if __name__ == "__main__":
    main()
