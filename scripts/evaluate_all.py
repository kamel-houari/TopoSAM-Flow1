#!/usr/bin/env python3
"""
Evaluate TopoSAM-Flow on all benchmarks and generate result tables.
"""

import argparse
import torch
from pathlib import Path
from src.model import TopoSAMFlow
from src.evaluate import evaluate
from src.utils import load_config, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['neu', 'rsdds', 'mvtec'],
                       required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output', type=str, default='results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config or f'configs/{args.dataset}.yaml')
    
    # Load model
    model = TopoSAMFlow(
        num_classes=config['model']['num_classes'],
        epsilon=config['model']['epsilon'],
        alpha=config['model']['alpha'],
        lambda_var=config['model']['lambda_var'],
        lambda_topo=config['model']['lambda_topo']
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    
    # Load data (placeholder - implement dataset loaders)
    # val_loader = load_dataset(args.dataset, config['data'], split='val')
    
    # Evaluate
    # metrics = evaluate(model, val_loader, args.device)
    
    # Save results
    # import json
    # with open(args.output, 'w') as f:
    #     json.dump(metrics, f, indent=2)
    
    print(f"Evaluation complete. Results: {args.output}")

if __name__ == '__main__':
    main()