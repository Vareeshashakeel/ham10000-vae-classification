"""
Central configuration for the HAM10000 CNN-baseline vs VAE-augmented experiment.
All paths are relative to the project root so the code runs unchanged on
Kaggle (where the root is typically /kaggle/working) or locally.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("HAM10000_PROJECT_ROOT", os.getcwd())

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data", "manifests")
DATA_SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")

OUTPUT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
OUTPUT_LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
OUTPUT_FIGURE_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUTPUT_PREDICTION_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")

# On Kaggle, the HAM10000 Kaggle mirror is mounted read-only here once the
# dataset is attached to the notebook:
#   /kaggle/input/skin-cancer-mnist-ham10000/
# Set this env var to override for local / other environments.
KAGGLE_INPUT_DIR = os.environ.get(
    "HAM10000_KAGGLE_INPUT",
    "/kaggle/input/skin-cancer-mnist-ham10000",
)

for _d in [
    DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_MANIFEST_DIR, DATA_SYNTHETIC_DIR,
    OUTPUT_CHECKPOINT_DIR, OUTPUT_LOG_DIR, OUTPUT_FIGURE_DIR, OUTPUT_PREDICTION_DIR,
]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes (HAM10000 diagnostic categories)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)

# Historically reported full-dataset counts (Tschandl et al. 2018). These are
# NOT hard-coded into the pipeline logic -- audit_data.py recomputes the real
# counts from the downloaded metadata CSV and everything downstream uses that.
REFERENCE_CLASS_COUNTS = {
    "nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514,
    "akiec": 327, "vasc": 142, "df": 115,
}
MINORITY_CLASSES = ["mel", "bkl", "bcc", "akiec", "vasc", "df"]  # everything but nv

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEEDS = [17, 29, 41]

# Adapts to whatever machine this runs on (Kaggle gives 2-4 CPUs typically;
# avoids the "excessive worker creation" warning on constrained environments).
NUM_WORKERS = max(0, min(2, (os.cpu_count() or 1) - 1))

# ---------------------------------------------------------------------------
# Split ratios (grouped by lesion_id, see make_group_splits.py)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---------------------------------------------------------------------------
# Classifier (DenseNet-121) hyperparameters
# ---------------------------------------------------------------------------
CLF_IMAGE_SIZE = 224
CLF_BATCH_SIZE = 32
CLF_DROPOUT = 0.30
CLF_LABEL_SMOOTHING = 0.05
CLF_HEAD_LR = 3e-4
CLF_BACKBONE_LR = 3e-5
CLF_WEIGHT_DECAY = 1e-4
CLF_WARMUP_EPOCHS = 3
CLF_STAGE_A_EPOCHS = 3     # frozen backbone, head only
CLF_STAGE_B_MAX_EPOCHS = 30  # unfrozen fine-tuning, early stopping on val macro-F1
CLF_EARLY_STOP_PATIENCE = 6

# ---------------------------------------------------------------------------
# VAE hyperparameters
# ---------------------------------------------------------------------------
VAE_IMAGE_SIZE = 128
VAE_LATENT_DIM = 128
VAE_BATCH_SIZE = 32
VAE_LR = 2e-4
VAE_MAX_EPOCHS = 150
VAE_EARLY_STOP_PATIENCE = 15
VAE_BETA_START = 1e-4
VAE_BETA_END = 0.5   # lowered from 1.0: full KL weight over-penalizes and biases
                     # the decoder toward blurry "average" reconstructions,
                     # which is especially damaging on tiny classes (df, vasc)
                     # that already have little data to begin with.
VAE_BETA_WARMUP_EPOCHS = 20

# ---------------------------------------------------------------------------
# Balancing policy conditions
# ---------------------------------------------------------------------------
CONDITIONS = ["C0", "C1", "C2", "C3"]
CONDITION_DESCRIPTIONS = {
    "C0": "Original imbalanced real training set (baseline).",
    "C1": "Real images + classical oversampling / augmentation, no VAE.",
    "C2": "Real images + VAE synthetic images, each minority class raised "
          "to >=50% of the majority training count.",
    "C3": "Real images + VAE synthetic images, each class raised to the "
          "majority training count (fully equalized).",
}
