#!/usr/bin/env python3
"""
Test optimized production variants combining multiple working optimizations
"""

import torch
import numpy as np
import time
import json
import sys
from pathlib import Path
import copy

sys.path.insert(0, '/Users/ktretina/claude_dir/ProteinMPNN')

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

device = torch.device("mps")

def load_model(model_path, k_neighbors=48):
    """Load model with specified k_neighbors."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=128,
        edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=k_neighbors
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    return model

def benchmark_variant(model, pdb_path, batch_size=1, num_runs=20):
    """Benchmark with specified batch size."""
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]
    batch_clones = [copy.deepcopy(protein) for _ in range(batch_size)]
    
    X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
        batch_clones, device, None, None, None, None, None, None, ca_only=False
    )
    
    seq_length = int(mask[0].sum().item())
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            randn = torch.randn(chain_M.shape, device=device)
            _ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
            torch.mps.synchronize()
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.mps.synchronize()
            start = time.perf_counter()
            randn = torch.randn(chain_M.shape, device=device)
            _ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
            torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times)
    mean_time = np.mean(times)
    time_per_protein = mean_time / batch_size
    
    return {
        'length': seq_length,
        'batch_size': batch_size,
        'mean_time_ms': float(mean_time * 1000),
        'time_per_protein_ms': float(time_per_protein * 1000),
        'throughput': float(seq_length * batch_size / mean_time)
    }

pdb_path = '/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb'
model_path = '/Users/ktretina/claude_dir/ProteinMPNN/vanilla_model_weights/v_48_020.pt'

print("="*70)
print("OPTIMIZED VARIANTS - Production Ready Configurations")
print("="*70)

variants = {
    'baseline': {
        'name': 'Baseline (k=48, batch=1)',
        'k_neighbors': 48,
        'batch_size': 1
    },
    'fast': {
        'name': 'Fast (k=16, batch=1)',
        'k_neighbors': 16,
        'batch_size': 1
    },
    'balanced': {
        'name': 'Balanced (k=32, batch=1)',
        'k_neighbors': 32,
        'batch_size': 1
    },
    'throughput': {
        'name': 'High-Throughput (k=32, batch=4)',
        'k_neighbors': 32,
        'batch_size': 4
    },
    'ultra_fast': {
        'name': 'Ultra-Fast (k=16, batch=4)',
        'k_neighbors': 16,
        'batch_size': 4
    }
}

results = {'variants': {}}

for variant_key, config in variants.items():
    print(f"\n{'-'*70}")
    print(f"VARIANT: {config['name']}")
    print(f"{'-'*70}")
    
    model = load_model(model_path, k_neighbors=config['k_neighbors'])
    result = benchmark_variant(model, pdb_path, batch_size=config['batch_size'])
    
    results['variants'][variant_key] = {
        'name': config['name'],
        'k_neighbors': config['k_neighbors'],
        'batch_size': config['batch_size'],
        'result': result
    }
    
    print(f"Time per protein: {result['time_per_protein_ms']:.2f} ms")
    print(f"Throughput: {result['throughput']:.1f} res/sec")

# Calculate speedups
baseline_time = results['variants']['baseline']['result']['time_per_protein_ms']

print("\n" + "="*70)
print("SPEEDUP ANALYSIS vs Baseline")
print("="*70)
print(f"\n{'Variant':<25} {'Time/Protein':<15} {'Speedup':<12} {'Config':<20}")
print("-"*72)

for variant_key, data in results['variants'].items():
    time_per_protein = data['result']['time_per_protein_ms']
    speedup = baseline_time / time_per_protein
    config_str = f"k={data['k_neighbors']}, b={data['batch_size']}"
    
    print(f"{data['name']:<25} {time_per_protein:<15.2f} {speedup:<12.2f}x {config_str:<20}")

# Find best performer
best_variant = min(results['variants'].items(), 
                   key=lambda x: x[1]['result']['time_per_protein_ms'])

print(f"\n{'='*70}")
print(f"BEST PERFORMER: {best_variant[1]['name']}")
print(f"Speedup: {baseline_time / best_variant[1]['result']['time_per_protein_ms']:.2f}x")
print(f"{'='*70}")

Path('output').mkdir(exist_ok=True)
with open('output/optimized_variants.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: output/optimized_variants.json")
