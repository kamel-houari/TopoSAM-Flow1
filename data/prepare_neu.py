
### **`data/prepare_neu.py`**
#```python
#!/usr/bin/env python3
"""
Prepare NEU Surface Defect Dataset for TopoSAM-Flow.
"""

import argparse
import os
import json
from pathlib import Path
import requests
from tqdm import tqdm

def download_neu(output_dir, extract=True):
    """Download NEU dataset (placeholder - actual download requires manual step)."""
    print("NEU dataset download requires manual step:")
    print("1. Visit: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_dataset.html")
    print("2. Download 'NEU-DET.zip'")
    print("3. Extract to:", output_dir)
    
    # Create directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Placeholder: create expected structure
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'annotations').mkdir(exist_ok=True)
    (output_dir / 'splits').mkdir(exist_ok=True)
    
    print(f"✓ Directory structure created at {output_dir}")
    return output_dir

def create_annotations(images_dir, annotations_dir):
    """Create bounding box annotations from image filenames (placeholder)."""
    # In practice: parse original NEU annotation format
    # Here: generate dummy annotations for demonstration
    
    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 
               'rolled_in', 'scratches']
    
    annotations = {}
    for cls in classes:
        cls_dir = images_dir / cls
        if cls_dir.exists():
            for img_path in cls_dir.glob('*.jpg'):
                # Dummy bounding box (center 50% of image, size 30%)
                annotations[img_path.name] = {
                    'class': cls,
                    'bbox': [70, 70, 130, 130]  # [x1, y1, x2, y2] for 200x200 image
                }
    
    # Save annotations
    with open(annotations_dir / 'boxes.json', 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Created {len(annotations)} bounding box annotations")

def create_splits(output_dir, train_ratio=0.8, val_ratio=0.1):
    """Create train/val/test splits."""
    import random
    random.seed(42)
    
    images_dir = output_dir / 'images'
    splits_dir = output_dir / 'splits'
    
    # Collect all images
    all_images = list(images_dir.rglob('*.jpg'))
    random.shuffle(all_images)
    
    n = len(all_images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    splits = {
        'train': [img.name for img in all_images[:train_end]],
        'val': [img.name for img in all_images[train_end:val_end]],
        'test': [img.name for img in all_images[val_end:]]
    }
    
    # Save splits
    for split_name, images in splits.items():
        with open(splits_dir / f'{split_name}.txt', 'w') as f:
            f.write('\n'.join(images))
    
    print(f"✓ Created splits: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', 
                       help='Download dataset (requires manual step)')
    parser.add_argument('--output', type=str, default='./data/NEU',
                       help='Output directory')
    parser.add_argument('--images', type=str, default=None,
                       help='Path to already downloaded images')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.download:
        download_neu(output_dir)
    
    if args.images:
        images_dir = Path(args.images)
        create_annotations(images_dir, output_dir / 'annotations')
        create_splits(output_dir, train_ratio=0.8, val_ratio=0.1)
    
    print(f"\n✓ NEU dataset preparation complete")
    print(f"  Location: {output_dir.absolute()}")

if __name__ == '__main__':
    main()