#!/bin/bash
# Train TopoSAM-Flow on MVTec AD dataset

set -e

CONFIG="configs/mvtec.yaml"
OUTPUT_DIR="outputs/mvtec"

# Train on all categories
python -m src.train \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --seed 42 \
    --device cuda \
    --evaluate_per_category true

echo "MVTec AD training complete"