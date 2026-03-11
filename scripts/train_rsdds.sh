#!/bin/bash
# Train TopoSAM-Flow on RSDDs dataset with cross-validation

set -e

CONFIG="configs/rsdds.yaml"
OUTPUT_DIR="outputs/rsdds"
FOLDS=5

for fold in $(seq 1 $FOLDS); do
    echo "=== Fold $fold/$FOLDS ==="
    
    python -m src.train \
        --config $CONFIG \
        --output_dir "${OUTPUT_DIR}/fold_${fold}" \
        --fold $fold \
        --seed 42 \
        --device cuda
done

# Aggregate results
python scripts/reproduce_tables.py --dataset rsdds --aggregate_folds