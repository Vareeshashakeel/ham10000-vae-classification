"""
Phase 4: Split.

Creates fixed, group-stratified train/val/test manifests so that no
lesion (or patient, if available) appears in more than one partition.
Runs once per seed in config.SEEDS and writes:
    data/manifests/splits_seed<seed>.csv

Each manifest row: image_id, lesion_id, patient_id, dx, label_idx,
file_path, partition, seed.

Usage:
    python src/make_group_splits.py --audit_csv data/processed/audit.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (  # noqa: E402
    CLASS_TO_IDX, DATA_MANIFEST_DIR, SEEDS, TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)


def grouped_stratified_split(df: pd.DataFrame, group_col: str, label_col: str, seed: int,
                              train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    """
    Assigns every group (lesion/patient) entirely to one partition, while
    ensuring EVERY class is represented in train/val/test proportionally
    (a per-class stratified group split), not just the overall totals.

    Naive global-capacity greedy allocation is a trap here: it satisfies the
    overall train/val/test size ratio but can starve val/test of minority
    classes entirely (verified by the smoke test on synthetic dummy data --
    see project notes). Splitting group lists independently within each
    class, then concatenating, fixes this and is the standard approach
    (equivalent in spirit to sklearn's StratifiedGroupKFold).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    from collections import defaultdict

    rng = np.random.RandomState(seed)

    # One representative label per group: the group's majority label.
    group_label = (
        df.groupby(group_col)[label_col]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    class_to_groups = defaultdict(list)
    for g, cls in group_label.items():
        class_to_groups[cls].append(g)

    group_to_partition = {}
    warnings = []

    for cls, cls_groups in class_to_groups.items():
        cls_groups = list(cls_groups)
        rng.shuffle(cls_groups)
        n = len(cls_groups)

        if n < 3:
            # Too few lesions/patients to split into 3 non-empty partitions
            # for this class -- put everything in train and warn loudly.
            # (This can genuinely happen for extreme minority classes and
            # must be surfaced, not silently hidden.)
            for g in cls_groups:
                group_to_partition[g] = "train"
            warnings.append(
                f"Class '{cls}' has only {n} group(s) -- ALL assigned to train. "
                f"This class cannot be evaluated on val/test with this grouping; "
                f"consider using a coarser group_col or flagging this as a limitation."
            )
            continue

        n_test = max(1, int(round(test_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        # keep at least 1 in train too
        n_val = min(n_val, n - n_test - 1) if (n - n_test - 1) >= 1 else max(0, n - n_test - 1)
        n_train = n - n_test - n_val

        for g in cls_groups[:n_train]:
            group_to_partition[g] = "train"
        for g in cls_groups[n_train:n_train + n_val]:
            group_to_partition[g] = "val"
        for g in cls_groups[n_train + n_val:]:
            group_to_partition[g] = "test"

    for w in warnings:
        print(f"WARNING: {w}")

    out = df.copy()
    out["partition"] = out[group_col].map(group_to_partition)
    out["label_idx"] = out[label_col].map(CLASS_TO_IDX)
    out["seed"] = seed
    return out


def report_split(split_df: pd.DataFrame, label_col: str):
    print("Partition sizes (images):")
    print(split_df["partition"].value_counts())
    print("\nPer-partition class distribution:")
    print(pd.crosstab(split_df["partition"], split_df[label_col]))


def main():
    parser = argparse.ArgumentParser(description="Create group-stratified HAM10000 splits.")
    parser.add_argument("--audit_csv", default=os.path.join("data", "processed", "audit.csv"))
    parser.add_argument("--group_col", default="lesion_id",
                         help="Use 'patient_id' if that column is reliable in your metadata, "
                              "otherwise 'lesion_id' (default, always available).")
    args = parser.parse_args()

    audit_df = pd.read_csv(args.audit_csv)
    valid_df = audit_df[audit_df["file_exists"] & audit_df["label_valid"]].copy()
    if audit_df["is_exact_duplicate_file"].any():
        # Keep only the first occurrence of each exact-duplicate file so a
        # duplicated image can't land in two different partitions.
        valid_df = valid_df.sort_values("image_id").drop_duplicates(subset="md5", keep="first")
        print(f"Dropped exact-duplicate files, kept first occurrence of each. "
              f"Remaining rows: {len(valid_df)}")

    for seed in SEEDS:
        split_df = grouped_stratified_split(valid_df, group_col=args.group_col,
                                             label_col="dx", seed=seed)
        out_path = os.path.join(DATA_MANIFEST_DIR, f"splits_seed{seed}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"\n=== seed {seed} -> {out_path} ===")
        report_split(split_df, "dx")


if __name__ == "__main__":
    main()
