#!/bin/bash
# Orchestration helper. Run ONE phase per Kaggle session (12-hour limit),
# saving outputs as you go. See README.md for the full session-by-session plan.
#
# Usage:
#   bash run_pipeline.sh audit
#   bash run_pipeline.sh split
#   bash run_pipeline.sh baseline <seed>
#   bash run_pipeline.sh train_vaes <seed>
#   bash run_pipeline.sh generate <seed> <policy: C2|C3>
#   bash run_pipeline.sh retrain <seed> <condition: C1|C2|C3>
#   bash run_pipeline.sh evaluate <seed> <condition>
set -e

KAGGLE_INPUT="${HAM10000_KAGGLE_INPUT:-/kaggle/input/skin-cancer-mnist-ham10000}"
PHASE=$1

case "$PHASE" in
  audit)
    python src/audit_data.py \
      --metadata_csv "$KAGGLE_INPUT/HAM10000_metadata.csv" \
      --image_dirs "$KAGGLE_INPUT/HAM10000_images_part_1" "$KAGGLE_INPUT/HAM10000_images_part_2"
    ;;

  split)
    python src/make_group_splits.py --audit_csv data/processed/audit.csv --group_col lesion_id
    ;;

  baseline)
    SEED=$2
    python src/train_classifier.py --manifest "data/manifests/splits_seed${SEED}.csv" \
      --seed "$SEED" --condition C0
    ;;

  train_vaes)
    SEED=$2
    for CLS in mel bkl bcc akiec vasc df; do
      echo "=== Training VAE for class: $CLS (seed $SEED) ==="
      python src/train_vae.py --manifest "data/manifests/splits_seed${SEED}.csv" \
        --class_name "$CLS" --seed "$SEED"
    done
    ;;

  generate)
    SEED=$2
    POLICY=$3
    python src/generate_synthetic.py --manifest "data/manifests/splits_seed${SEED}.csv" \
      --seed "$SEED" --policy "$POLICY" --mode manifold
    ;;

  retrain)
    SEED=$2
    CONDITION=$3
    if [ "$CONDITION" == "C1" ]; then
      python src/train_classifier.py --manifest "data/manifests/splits_seed${SEED}.csv" \
        --seed "$SEED" --condition C1
    else
      python src/train_classifier.py --manifest "data/manifests/splits_seed${SEED}.csv" \
        --seed "$SEED" --condition "$CONDITION" \
        --synthetic_manifest "data/manifests/synthetic_seed${SEED}_${CONDITION}.csv" \
        --real_only_finetune_epochs 3
    fi
    ;;

  evaluate)
    SEED=$2
    CONDITION=$3
    python src/evaluate.py --manifest "data/manifests/splits_seed${SEED}.csv" \
      --checkpoint "outputs/checkpoints/densenet121_${CONDITION}_seed${SEED}.pt" \
      --condition "$CONDITION" --seed "$SEED"
    ;;

  *)
    echo "Unknown phase: $PHASE"
    echo "Valid phases: audit, split, baseline, train_vaes, generate, retrain, evaluate"
    exit 1
    ;;
esac
