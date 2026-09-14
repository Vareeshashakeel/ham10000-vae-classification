"""
Phase 8: Balance -- generate synthetic minority-class images from trained
VAEs, according to a balancing policy (C1 classical / C2 50%-cap / C3
equalized), and run basic automated quality checks.

Writes:
  data/synthetic/<class>_seed<seed>/synth_*.png
  data/manifests/synthetic_seed<seed>_<policy>.csv   (file_path, dx, source)
  outputs/predictions/synthetic_review_seed<seed>_<policy>.csv

Usage:
    python src/generate_synthetic.py --manifest data/manifests/splits_seed17.csv \
        --seed 17 --policy C3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import (  # noqa: E402
    CLASS_NAMES, DATA_MANIFEST_DIR, DATA_SYNTHETIC_DIR, OUTPUT_CHECKPOINT_DIR,
    OUTPUT_PREDICTION_DIR,
)
from models_vae import ConvVAE  # noqa: E402


def compute_train_counts(manifest_path: str):
    df = pd.read_csv(manifest_path)
    train_df = df[df["partition"] == "train"]
    counts = train_df["dx"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict()
    return counts, train_df


def targets_for_policy(train_counts: dict, policy: str):
    majority = max(train_counts.values())
    targets = {}
    for cls, n in train_counts.items():
        if policy == "C1":
            targets[cls] = n  # no VAE synthesis; classical oversampling handled separately
        elif policy == "C2":
            targets[cls] = max(n, int(round(0.5 * majority)))
        elif policy == "C3":
            targets[cls] = majority
        else:
            raise ValueError(f"Unknown policy: {policy}")
    return targets


def basic_quality_checks(img_array: np.ndarray):
    """
    Lightweight automated screen (Section 10 of the plan): flags obviously
    malformed images. This does NOT replace a manual visual review pass --
    it just catches degenerate decoder failures automatically.
    """
    flags = []
    if img_array.min() < -1e-3 or img_array.max() > 1 + 1e-3:
        flags.append("out_of_range")
    std = img_array.std()
    if std < 0.01:
        flags.append("near_blank_low_variance")
    return flags


def generate_for_class(class_name: str, n_to_generate: int, seed: int, device,
                        real_file_paths=None, mode: str = "manifold"):
    """
    mode='manifold' (default, recommended): generates each synthetic image
        by encoding TWO real training images of this class and decoding a
        random interpolation between their latent codes (plus a small noise
        nudge for variety). This keeps synthetic images anchored close to
        the real data manifold, rather than floating toward the "generic
        average" look that pure-prior sampling tends to produce -- that
        generic look is a likely source of the train/val gap seen in C3.
    mode='prior': the original behavior -- decode a random N(0,1) latent.
        Kept available for comparison / ablation.
    real_file_paths: required for mode='manifold' -- list of real training
        image paths for this class.
    """
    if n_to_generate <= 0:
        return []

    ckpt_path = os.path.join(OUTPUT_CHECKPOINT_DIR, f"vae_{class_name}_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No trained VAE checkpoint for class '{class_name}' seed {seed} at {ckpt_path}. "
            f"Run train_vae.py for this class/seed first."
        )

    model = ConvVAE().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    out_dir = os.path.join(DATA_SYNTHETIC_DIR, f"{class_name}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed * 1000 + hash(class_name) % 1000)
    rng = np.random.RandomState(seed * 1000 + hash(class_name) % 1000)
    records = []
    latent_seed_counter = 0

    if mode == "manifold":
        if not real_file_paths or len(real_file_paths) < 2:
            raise ValueError(
                f"mode='manifold' needs at least 2 real training images for class "
                f"'{class_name}', got {len(real_file_paths) if real_file_paths else 0}. "
                f"Falling back to mode='prior' for this class is a reasonable alternative."
            )
        from datasets import build_vae_transforms
        transform = build_vae_transforms(train=False)

        # Pre-encode all real images once (cheap relative to generation loop).
        with torch.no_grad():
            real_tensors = torch.stack([
                transform(Image.open(p).convert("RGB")) for p in real_file_paths
            ]).to(device)
            mus, _ = model.encode(real_tensors)  # use mu only -- deterministic, cleaner anchor

        n_real = mus.size(0)
        batch_size = 32
        generated = 0
        while generated < n_to_generate:
            n = min(batch_size, n_to_generate - generated)
            idx_a = rng.randint(0, n_real, size=n)
            idx_b = rng.randint(0, n_real, size=n)
            alpha = torch.tensor(rng.uniform(0.3, 0.7, size=n), dtype=torch.float32, device=device).unsqueeze(1)
            z = alpha * mus[idx_a] + (1 - alpha) * mus[idx_b]
            z = z + 0.05 * torch.randn_like(z)  # small nudge so it's not just a deterministic blend
            with torch.no_grad():
                samples = model.decode(z).clamp(0.0, 1.0).cpu().numpy()

            for i in range(n):
                img_arr = samples[i]
                flags = basic_quality_checks(img_arr)
                img_uint8 = (np.clip(img_arr, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
                img_id = f"synth_{class_name}_seed{seed}_{latent_seed_counter:06d}"
                file_path = os.path.join(out_dir, img_id + ".png")
                Image.fromarray(img_uint8).save(file_path)
                records.append({
                    "image_id": img_id, "file_path": file_path, "dx": class_name,
                    "source": "vae_synthetic_manifold", "source_vae_checkpoint": ckpt_path,
                    "latent_seed": latent_seed_counter, "seed": seed,
                    "review_status": "rejected" if flags else "unreviewed",
                    "auto_flags": ";".join(flags),
                })
                latent_seed_counter += 1
            generated += n
        return records

    # mode == "prior": original behavior
    batch_size = 32
    generated = 0
    while generated < n_to_generate:
        n = min(batch_size, n_to_generate - generated)
        with torch.no_grad():
            samples = model.sample(n, device=device).cpu().numpy()
        for i in range(n):
            img_arr = samples[i]
            flags = basic_quality_checks(img_arr)
            img_uint8 = (np.clip(img_arr, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
            img_id = f"synth_{class_name}_seed{seed}_{latent_seed_counter:06d}"
            file_path = os.path.join(out_dir, img_id + ".png")
            Image.fromarray(img_uint8).save(file_path)
            records.append({
                "image_id": img_id, "file_path": file_path, "dx": class_name,
                "source": "vae_synthetic_prior", "source_vae_checkpoint": ckpt_path,
                "latent_seed": latent_seed_counter, "seed": seed,
                "review_status": "rejected" if flags else "unreviewed",
                "auto_flags": ";".join(flags),
            })
            latent_seed_counter += 1
        generated += n

    return records


def main():
    parser = argparse.ArgumentParser(description="Generate VAE synthetic images per balancing policy.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", choices=["C2", "C3"], required=True,
                         help="C1 uses classical oversampling, not VAE generation -- handle "
                              "that directly in train_classifier.py's sampler instead.")
    parser.add_argument("--mode", choices=["manifold", "prior"], default="manifold",
                         help="'manifold' (default): interpolate between real images' latents -- "
                              "stays closer to the real data manifold. 'prior': original random-"
                              "latent sampling, kept for comparison.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_counts, train_df = compute_train_counts(args.manifest)
    targets = targets_for_policy(train_counts, args.policy)

    print(f"Train counts: {train_counts}")
    print(f"Targets ({args.policy}): {targets}")
    print(f"Generation mode: {args.mode}")

    all_records = []
    for cls in CLASS_NAMES:
        n_needed = max(0, targets[cls] - train_counts[cls])
        if n_needed == 0:
            continue
        print(f"Generating {n_needed} synthetic images for class '{cls}'...")
        real_paths = train_df[train_df["dx"] == cls]["file_path"].tolist() if args.mode == "manifold" else None
        recs = generate_for_class(cls, n_needed, args.seed, device,
                                   real_file_paths=real_paths, mode=args.mode)
        all_records.extend(recs)

    synth_df = pd.DataFrame(all_records)
    manifest_out = os.path.join(DATA_MANIFEST_DIR, f"synthetic_seed{args.seed}_{args.policy}.csv")
    synth_df.to_csv(manifest_out, index=False)

    review_out = os.path.join(OUTPUT_PREDICTION_DIR, f"synthetic_review_seed{args.seed}_{args.policy}.csv")
    synth_df.to_csv(review_out, index=False)

    n_flagged = (synth_df["review_status"] == "rejected").sum() if len(synth_df) else 0
    print(f"Generated {len(synth_df)} synthetic images total. "
          f"{n_flagged} auto-flagged as degenerate. Manifest: {manifest_out}")
    print("NOTE: 'unreviewed' images still need the manual visual-plausibility pass "
          "described in Section 10 of the plan before being trusted at scale.")


if __name__ == "__main__":
    main()
