"""
Phase 3: Audit.

Verifies the downloaded HAM10000 metadata + images before anything is
fitted. Produces:
  - data/processed/audit.csv          (per-row validity flags)
  - data/processed/class_distribution.png
  - prints a summary report to stdout

Usage (on Kaggle, after attaching the "skin-cancer-mnist-ham10000" dataset):
    python src/audit_data.py \
        --metadata_csv /kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv \
        --image_dirs /kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1 \
                     /kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2
"""
import argparse
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, DATA_PROCESSED_DIR, OUTPUT_FIGURE_DIR  # noqa: E402


def find_image_path(image_id: str, image_dirs):
    """HAM10000 ships images across two folders; locate whichever has it."""
    for d in image_dirs:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(d, image_id + ext)
            if os.path.isfile(candidate):
                return candidate
    return None


def file_md5(path, chunk_size=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def run_audit(metadata_csv: str, image_dirs, compute_hashes: bool = True):
    df = pd.read_csv(metadata_csv)

    required_cols = {"image_id", "dx", "lesion_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Metadata CSV is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    has_patient_id = "patient_id" in df.columns or "lesion_id" in df.columns
    group_col = "patient_id" if "patient_id" in df.columns else "lesion_id"

    rows = []
    for _, row in df.iterrows():
        image_id = row["image_id"]
        label = row["dx"]
        lesion_id = row["lesion_id"]

        path = find_image_path(image_id, image_dirs)
        file_exists = path is not None
        label_valid = label in CLASS_NAMES

        rows.append({
            "image_id": image_id,
            "lesion_id": lesion_id,
            "patient_id": row.get("patient_id", lesion_id),
            "dx": label,
            "file_exists": file_exists,
            "label_valid": label_valid,
            "file_path": path,
            "md5": file_md5(path) if (compute_hashes and file_exists) else None,
        })

    audit_df = pd.DataFrame(rows)

    # Duplicate detection: same md5 hash appearing more than once.
    if compute_hashes:
        dup_counts = audit_df["md5"].value_counts()
        dup_hashes = dup_counts[dup_counts > 1].index
        audit_df["is_exact_duplicate_file"] = audit_df["md5"].isin(dup_hashes)
    else:
        audit_df["is_exact_duplicate_file"] = False

    audit_csv_path = os.path.join(DATA_PROCESSED_DIR, "audit.csv")
    audit_df.to_csv(audit_csv_path, index=False)

    # ---- Summary report ----
    n_total = len(audit_df)
    n_missing_files = (~audit_df["file_exists"]).sum()
    n_invalid_labels = (~audit_df["label_valid"]).sum()
    n_exact_dupes = audit_df["is_exact_duplicate_file"].sum()
    n_unique_lesions = audit_df["lesion_id"].nunique()
    n_unique_patients = audit_df["patient_id"].nunique() if "patient_id" in audit_df else n_unique_lesions
    multi_image_lesions = (audit_df.groupby("lesion_id").size() > 1).sum()

    class_counts = audit_df[audit_df["label_valid"]]["dx"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)

    print("=" * 70)
    print("HAM10000 DATA AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total metadata rows:            {n_total}")
    print(f"Missing image files:            {n_missing_files}")
    print(f"Invalid / unexpected labels:    {n_invalid_labels}")
    print(f"Exact duplicate files (by MD5): {n_exact_dupes}")
    print(f"Unique lesion_ids:              {n_unique_lesions}")
    print(f"Unique patient/group ids:       {n_unique_patients}  (grouping column: '{group_col}')")
    print(f"Lesions with >1 image:          {multi_image_lesions}  <-- leakage risk if split ignores this")
    print("-" * 70)
    print("Recomputed class distribution (from metadata, not hard-coded):")
    for c in CLASS_NAMES:
        print(f"  {c:6s}: {class_counts[c]:5d}")
    print("=" * 70)

    if n_missing_files > 0:
        print(f"WARNING: {n_missing_files} images referenced in metadata were not found "
              f"on disk. Check --image_dirs paths.")
    if multi_image_lesions > 0:
        print(f"NOTE: {multi_image_lesions} lesions have multiple images. "
              f"make_group_splits.py MUST split by '{group_col}', never by image_id, "
              f"to avoid train/test leakage.")

    # Class distribution plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        class_counts.plot(kind="bar")
        plt.title("HAM10000 class distribution (recomputed from metadata)")
        plt.ylabel("Image count")
        plt.xlabel("Diagnostic class")
        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_FIGURE_DIR, "class_distribution.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved class distribution plot to: {fig_path}")
    except ImportError:
        print("matplotlib not available; skipped class distribution plot.")

    print(f"Saved full audit table to: {audit_csv_path}")
    return audit_df, group_col


def main():
    parser = argparse.ArgumentParser(description="Audit HAM10000 metadata + images.")
    parser.add_argument("--metadata_csv", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--image_dirs", nargs="+", required=True,
                         help="One or more directories containing the .jpg images")
    parser.add_argument("--no_hashes", action="store_true",
                         help="Skip MD5 hashing (faster, but no duplicate detection)")
    args = parser.parse_args()

    run_audit(args.metadata_csv, args.image_dirs, compute_hashes=not args.no_hashes)


if __name__ == "__main__":
    main()
