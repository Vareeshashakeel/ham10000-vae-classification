"""
Phase 6: Train one class-specific VAE.

Trains only on TRAIN-partition images of a single class (never val/test --
the frozen test/val partitions must never touch VAE training). Saves:
  outputs/checkpoints/vae_<class>_seed<seed>.pt
  outputs/figures/vae_<class>_seed<seed>_reconstructions.png
  outputs/figures/vae_<class>_seed<seed>_samples.png
  outputs/logs/vae_<class>_seed<seed>_losses.csv

Usage:
    python src/train_vae.py --manifest data/manifests/splits_seed17.csv \
        --class_name df --seed 17
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import NUM_WORKERS  # noqa: E402
from config import (  # noqa: E402
    OUTPUT_CHECKPOINT_DIR, OUTPUT_FIGURE_DIR, OUTPUT_LOG_DIR,
    VAE_BATCH_SIZE, VAE_BETA_END, VAE_BETA_START, VAE_BETA_WARMUP_EPOCHS,
    VAE_EARLY_STOP_PATIENCE, VAE_LR, VAE_MAX_EPOCHS,
)
from datasets import SingleClassImageDataset  # noqa: E402
from models_vae import ConvVAE, beta_schedule, vae_loss  # noqa: E402


def save_image_grid(tensor_batch, path, nrow=5):
    from torchvision.utils import save_image
    save_image(tensor_batch, path, nrow=nrow)


def train_one_vae(manifest_path: str, class_name: str, seed: int,
                   max_epochs: int = VAE_MAX_EPOCHS, device=None):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(manifest_path)
    train_df = df[(df["partition"] == "train") & (df["dx"] == class_name)]
    if len(train_df) < 10:
        raise ValueError(
            f"Only {len(train_df)} training images for class '{class_name}' -- "
            f"too few to train a VAE meaningfully. Check the manifest."
        )

    file_paths = train_df["file_path"].tolist()

    # Split by file path first (not via random_split of one Dataset) so
    # train and val get DIFFERENT transforms: train gets flip/rotation
    # augmentation, the small internal val set stays un-augmented for a
    # clean, comparable reconstruction-quality signal.
    rng_split = np.random.RandomState(seed)
    shuffled_paths = list(file_paths)
    rng_split.shuffle(shuffled_paths)
    n_val = max(1, int(0.1 * len(shuffled_paths)))
    val_paths = shuffled_paths[:n_val]
    train_paths = shuffled_paths[n_val:]

    train_ds = SingleClassImageDataset(train_paths, train=True)
    val_ds = SingleClassImageDataset(val_paths, train=False)

    train_loader = DataLoader(train_ds, batch_size=VAE_BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, drop_last=len(train_ds) > VAE_BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=VAE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ConvVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    ckpt_path = os.path.join(OUTPUT_CHECKPOINT_DIR, f"vae_{class_name}_seed{seed}.pt")

    for epoch in range(1, max_epochs + 1):
        beta = beta_schedule(epoch, VAE_BETA_WARMUP_EPOCHS, VAE_BETA_START, VAE_BETA_END)

        model.train()
        train_loss_sum, train_recon_sum, train_kl_sum, n_batches = 0.0, 0.0, 0.0, 0
        for x in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_recon_sum += recon.item()
            train_kl_sum += kl.item()
            n_batches += 1
        train_loss = train_loss_sum / max(1, n_batches)

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                loss, _, _ = vae_loss(x_hat, x, mu, logvar, beta)
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(1, val_batches)

        history.append({
            "epoch": epoch, "beta": beta, "train_loss": train_loss,
            "train_recon": train_recon_sum / max(1, n_batches),
            "train_kl": train_kl_sum / max(1, n_batches), "val_loss": val_loss,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{class_name} seed{seed}] epoch {epoch:3d}/{max_epochs} "
                  f"beta={beta:.4f} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "class_name": class_name, "seed": seed}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= VAE_EARLY_STOP_PATIENCE:
                print(f"[{class_name} seed{seed}] early stopping at epoch {epoch}")
                break

    # Save loss curve
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(OUTPUT_LOG_DIR, f"vae_{class_name}_seed{seed}_losses.csv")
    hist_df.to_csv(hist_csv, index=False)

    # Reload best checkpoint and save qualitative panels
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    model.eval()

    with torch.no_grad():
        batch = next(iter(val_loader)).to(device)
        n_show = min(5, batch.size(0))
        x_hat, _, _ = model(batch[:n_show])
        recon_grid = torch.cat([batch[:n_show], x_hat[:n_show]], dim=0)
        save_image_grid(recon_grid, os.path.join(
            OUTPUT_FIGURE_DIR, f"vae_{class_name}_seed{seed}_reconstructions.png"), nrow=n_show)

        samples = model.sample(25, device=device)
        save_image_grid(samples, os.path.join(
            OUTPUT_FIGURE_DIR, f"vae_{class_name}_seed{seed}_samples.png"), nrow=5)

    print(f"[{class_name} seed{seed}] done. best_val_loss={best_val_loss:.4f}. "
          f"Checkpoint: {ckpt_path}")
    return ckpt_path, hist_csv


def main():
    parser = argparse.ArgumentParser(description="Train a class-specific VAE.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_epochs", type=int, default=VAE_MAX_EPOCHS)
    args = parser.parse_args()
    train_one_vae(args.manifest, args.class_name, args.seed, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
