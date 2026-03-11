#!/bin/bash
# Train TopoSAM-Flow on NEU dataset

set -e

# Configuration
CONFIG="configs/neu.yaml"
OUTPUT_DIR="outputs/neu"
SEED=42

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python -m src.train \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --device cuda \
    --sam_checkpoint "checkpoints/sam_vit_b_01ec64.pth"

echo "Training complete. Results saved to $OUTPUT_DIR"