#!/usr/bin/env python3
"""
Benchmark EXTREME-v2: Testing k=12 with optimized variants
Building on 6.85x speedup to achieve 7.5x+ speedup
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

def load_model(model_path, num_encoder_layers=3, num_decoder_layers=3,
               hidden_dim=128, k_neighbors=48):
    """Load model with configurable architecture."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = ProteinMPNN(
        ca_only=False, num_letters=21,
        node_features=hidden_dim, edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        augment_eps=0.0, k_neighbors=k_neighbors
    )

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    model.to(device)
    model.eval()
    return model

def benchmark_variant(model, pdb_path, batch_size=1, num_runs=20):
    """Benchmark with batching."""
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
print("EXTREME-v2 VARIANTS - Testing k=12 Optimization")
print("="*70)

variants = {
    'baseline': {
        'name': 'Baseline',
        'encoder': 3, 'decoder': 3, 'dim': 128, 'k': 48, 'batch': 1
    },
    'extreme_original': {
        'name': 'EXTREME (k=16)',
        'encoder': 2, 'decoder': 2, 'dim': 64, 'k': 16, 'batch': 8
    },
    'extreme_v2': {
        'name': 'EXTREME-v2 (k=12)',
        'encoder': 2, 'decoder': 2, 'dim': 64, 'k': 12, 'batch': 8
    },
    'extreme_v2_batch4': {
        'name': 'EXTREME-v2 (k=12, batch=4)',
        'encoder': 2, 'decoder': 2, 'dim': 64, 'k': 12, 'batch': 4
    },
    'minimal_k12': {
        'name': 'Minimal+Fast (k=12)',
        'encoder': 2, 'decoder': 2, 'dim': 64, 'k': 12, 'batch': 1
    },
}

results = {'variants': {}}

for variant_key, config in variants.items():
    print(f"\n{'-'*70}")
    print(f"VARIANT: {config['name']}")
    print(f"Config: {config['encoder']}+{config['decoder']} layers, dim={config['dim']}, k={config['k']}, batch={config['batch']}")
    print(f"{'-'*70}")

    model = load_model(
        model_path,
        num_encoder_layers=config['encoder'],
        num_decoder_layers=config['decoder'],
        hidden_dim=config['dim'],
        k_neighbors=config['k']
    )

    result = benchmark_variant(model, pdb_path, batch_size=config['batch'])

    results['variants'][variant_key] = {
        'name': config['name'],
        'config': config,
        'result': result
    }

    print(f"Time per protein: {result['time_per_protein_ms']:.2f} ms")
    print(f"Throughput: {result['throughput']:.1f} res/sec")

# Analysis
baseline_time = results['variants']['baseline']['result']['time_per_protein_ms']

print("\n" + "="*70)
print("EXTREME-v2 PERFORMANCE COMPARISON")
print("="*70)
print(f"\n{'Variant':<35} {'Time/Protein':<15} {'Speedup':<12} {'Throughput':<15}")
print("-"*77)

for variant_key, data in results['variants'].items():
    result = data['result']
    speedup = baseline_time / result['time_per_protein_ms']
    print(f"{data['name']:<35} "
          f"{result['time_per_protein_ms']:<15.2f} "
          f"{speedup:<12.2f}x "
          f"{result['throughput']:<15.1f}")

# Compare k=16 vs k=12
extreme_orig = results['variants']['extreme_original']['result']['time_per_protein_ms']
extreme_v2 = results['variants']['extreme_v2']['result']['time_per_protein_ms']
improvement = (extreme_orig - extreme_v2) / extreme_orig * 100

print(f"\n{'='*70}")
print(f"k=12 vs k=16 IMPROVEMENT ANALYSIS")
print(f"{'='*70}")
print(f"\nEXTREME (k=16):    {extreme_orig:.2f} ms/protein")
print(f"EXTREME-v2 (k=12): {extreme_v2:.2f} ms/protein")
print(f"Improvement:       {improvement:+.1f}%")

if improvement > 5:
    print("\n✅ Significant improvement! k=12 is viable.")
    print("   Recommendation: Validate accuracy on your test set")
else:
    print("\n⚠️  Marginal improvement. k=16 may be safer without validation.")

# Find best overall
best_variant = min(results['variants'].items(),
                   key=lambda x: x[1]['result']['time_per_protein_ms'])

print(f"\n{'='*70}")
print(f"BEST PERFORMER")
print(f"{'='*70}")
print(f"Variant: {best_variant[1]['name']}")
print(f"Speedup: {baseline_time / best_variant[1]['result']['time_per_protein_ms']:.2f}x")
print(f"Time: {best_variant[1]['result']['time_per_protein_ms']:.2f} ms/protein")
print(f"Throughput: {best_variant[1]['result']['throughput']:.1f} res/sec")

print(f"\n{'='*70}")
print(f"RECOMMENDATIONS")
print(f"{'='*70}")

speedup_v2 = baseline_time / extreme_v2

if speedup_v2 > 7.0:
    print(f"\n🎉 EXTREME-v2 achieved {speedup_v2:.2f}x speedup!")
    print("   This surpasses the 7.0x milestone.")
    print("\n   Next steps:")
    print("   1. Validate accuracy on your protein design tasks")
    print("   2. If accuracy is acceptable, use EXTREME-v2 in production")
    print("   3. Proceed to knowledge distillation for 10-15x speedup")
elif speedup_v2 > 6.85:
    print(f"\n✅ EXTREME-v2 achieved {speedup_v2:.2f}x speedup")
    print("   Small improvement over EXTREME (6.85x)")
    print("\n   Recommendation:")
    print("   - Test accuracy on validation set")
    print("   - If accuracy loss <5%, adopt EXTREME-v2")
    print("   - Otherwise, stick with EXTREME (k=16)")
else:
    print(f"\n⚠️  EXTREME-v2 achieved {speedup_v2:.2f}x speedup")
    print("   No improvement over EXTREME (6.85x)")
    print("   Recommendation: Stick with k=16")

Path('output').mkdir(exist_ok=True)
with open('output/extreme_v2_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("Results saved to: output/extreme_v2_benchmarks.json")
print("="*70)
