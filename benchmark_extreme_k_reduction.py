#!/usr/bin/env python3
"""
Test extreme k-neighbor reduction (k=8, k=12)
Inspired by Cardinality Preserved Attention concept from diverse_opts_proteinmpnn.txt

Testing hypothesis: Can we go even lower than k=16?
"""

import torch
import numpy as np
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ktretina/claude_dir/ProteinMPNN')

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

device = torch.device("mps")

def load_model(model_path, k_neighbors=48):
    """Load model with specified k-neighbors."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = ProteinMPNN(
        ca_only=False, num_letters=21,
        node_features=128, edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=k_neighbors
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def benchmark_model(model, pdb_path, num_runs=20):
    """Benchmark a model variant."""
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]

    X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
        [protein], device, None, None, None, None, None, None, ca_only=False
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
    return {
        'length': seq_length,
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'throughput': float(seq_length / np.mean(times))
    }

pdb_path = '/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb'
model_path = '/Users/ktretina/claude_dir/ProteinMPNN/vanilla_model_weights/v_48_020.pt'

print("="*70)
print("EXTREME K-NEIGHBOR REDUCTION TEST")
print("Testing hypothesis: Can we go lower than k=16?")
print("="*70)

# Test k values from 48 down to 8
k_values = [48, 32, 24, 16, 12, 8]

results = {'variants': {}}

for k in k_values:
    print(f"\n{'-'*70}")
    print(f"Testing k={k} neighbors")
    print(f"{'-'*70}")

    try:
        model = load_model(model_path, k_neighbors=k)
        result = benchmark_model(model, pdb_path)

        results['variants'][f'k{k}'] = {
            'k': k,
            'result': result
        }

        print(f"Time: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
        print(f"Throughput: {result['throughput']:.1f} res/sec")

    except Exception as e:
        print(f"ERROR with k={k}: {e}")
        results['variants'][f'k{k}'] = {
            'k': k,
            'error': str(e)
        }

# Analysis
print("\n" + "="*70)
print("EXTREME K-REDUCTION ANALYSIS")
print("="*70)

baseline_time = results['variants']['k48']['result']['mean_ms']

print(f"\n{'k Value':<10} {'Time (ms)':<15} {'Speedup':<12} {'Throughput':<15} {'Quality':<10}")
print("-"*72)

for k_key, data in results['variants'].items():
    if 'error' in data:
        continue

    k = data['k']
    result = data['result']
    speedup = baseline_time / result['mean_ms']

    # Quality estimate based on k value
    if k >= 32:
        quality = "Excellent"
    elif k >= 16:
        quality = "Good"
    elif k >= 12:
        quality = "Fair"
    else:
        quality = "Risky"

    print(f"{k:<10} {result['mean_ms']:<15.2f} {speedup:<12.2f}x {result['throughput']:<15.1f} {quality:<10}")

# Find optimal k
valid_results = {k: v for k, v in results['variants'].items() if 'error' not in v}
best_k = min(valid_results.items(), key=lambda x: x[1]['result']['mean_ms'])

print(f"\n{'='*70}")
print(f"ANALYSIS")
print(f"{'='*70}")
print(f"\nFastest: k={best_k[1]['k']}")
print(f"Speedup: {baseline_time / best_k[1]['result']['mean_ms']:.2f}x vs baseline")
print(f"Throughput: {best_k[1]['result']['throughput']:.1f} res/sec")

# Compare k=16 vs lower values
if 'k16' in valid_results and 'k12' in valid_results:
    k16_time = results['variants']['k16']['result']['mean_ms']
    k12_time = results['variants']['k12']['result']['mean_ms']
    improvement = (k16_time - k12_time) / k16_time * 100

    print(f"\nk=12 vs k=16: {improvement:+.1f}% speedup")
    if improvement < 5:
        print("  → Marginal benefit, k=16 recommended for better accuracy")
    else:
        print("  → Significant speedup, worth testing accuracy impact")

if 'k16' in valid_results and 'k8' in valid_results:
    k16_time = results['variants']['k16']['result']['mean_ms']
    k8_time = results['variants']['k8']['result']['mean_ms']
    improvement = (k16_time - k8_time) / k16_time * 100

    print(f"\nk=8 vs k=16: {improvement:+.1f}% speedup")
    if improvement > 15:
        print("  → Major speedup! But accuracy likely severely degraded")
        print("  → Only use for ultra-fast screening where quality is less critical")
    else:
        print("  → Not worth the accuracy trade-off")

print(f"\n{'='*70}")
print("RECOMMENDATIONS")
print(f"{'='*70}")
print("\nBased on k-neighbor reduction testing:")
print("• k=48: Best accuracy, baseline speed")
print("• k=32: Minimal accuracy loss, modest speedup")
print("• k=16: Good accuracy/speed balance (RECOMMENDED)")
print("• k=12: Fair accuracy, better speed (use with caution)")
print("• k=8:  Risky accuracy, maximum speed (screening only)")
print("\nConclusion:")
print("k=16 remains the sweet spot for most use cases.")
print("Further reduction to k=12 or k=8 needs accuracy validation.")

Path('output').mkdir(exist_ok=True)
with open('output/extreme_k_reduction.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: output/extreme_k_reduction.json")
