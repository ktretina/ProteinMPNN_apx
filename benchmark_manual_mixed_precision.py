#!/usr/bin/env python3
"""
Test Manual Mixed Precision Optimization
Cast Linear layer weights to FP16 while keeping activations in FP32

From expert_proteinmpnn.txt optimization #4:
- Weights: Cast to float16
- Bias: Keep float32
- Input/Output Activations: Keep float32
- Matrix multiplication uses mixed precision (FP16 weights, FP32 accumulation)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ktretina/claude_dir/ProteinMPNN')

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

device = torch.device("mps")

def apply_manual_mixed_precision(model):
    """
    Apply selective mixed precision to Linear layers only.

    Strategy:
    - Cast Linear layer weights to FP16
    - Keep biases in FP32
    - Keep all activations in FP32
    - Let MPS handle mixed precision matmul
    """
    converted_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Cast weights to FP16
            if module.weight is not None:
                module.weight.data = module.weight.data.half()
                converted_count += 1

            # Keep bias in FP32 (if it exists)
            # Bias stays as-is (FP32)

    print(f"Converted {converted_count} Linear layer weights to FP16")
    return model

def load_model_fp32(model_path):
    """Load baseline FP32 model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = ProteinMPNN(
        ca_only=False, num_letters=21,
        node_features=128, edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=48
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_model_mixed_precision(model_path):
    """Load model with manual mixed precision."""
    model = load_model_fp32(model_path)
    model = apply_manual_mixed_precision(model)
    return model

def benchmark_model(model, pdb_path, num_runs=20, name="Model"):
    """Benchmark a model variant."""

    print(f"\nBenchmarking {name}...")

    try:
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
            'success': True,
            'length': seq_length,
            'mean_ms': float(np.mean(times) * 1000),
            'std_ms': float(np.std(times) * 1000),
            'throughput': float(seq_length / np.mean(times))
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'success': False,
            'error': str(e)
        }

pdb_path = '/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb'
model_path = '/Users/ktretina/claude_dir/ProteinMPNN/vanilla_model_weights/v_48_020.pt'

print("="*70)
print("MANUAL MIXED PRECISION OPTIMIZATION TEST")
print("Expert optimization #4: FP16 weights, FP32 activations")
print("="*70)

# Test 1: Baseline FP32
print("\n" + "-"*70)
print("1. BASELINE (Full FP32)")
print("-"*70)

model_fp32 = load_model_fp32(model_path)
result_fp32 = benchmark_model(model_fp32, pdb_path, name="FP32 Baseline")

if result_fp32['success']:
    print(f"Time: {result_fp32['mean_ms']:.2f} ± {result_fp32['std_ms']:.2f} ms")
    print(f"Throughput: {result_fp32['throughput']:.1f} res/sec")

# Test 2: Manual Mixed Precision
print("\n" + "-"*70)
print("2. MANUAL MIXED PRECISION (FP16 weights, FP32 activations)")
print("-"*70)

model_mixed = load_model_mixed_precision(model_path)
result_mixed = benchmark_model(model_mixed, pdb_path, name="Mixed Precision")

if result_mixed['success']:
    print(f"Time: {result_mixed['mean_ms']:.2f} ± {result_mixed['std_ms']:.2f} ms")
    print(f"Throughput: {result_mixed['throughput']:.1f} res/sec")

    if result_fp32['success']:
        speedup = result_fp32['mean_ms'] / result_mixed['mean_ms']
        print(f"Speedup: {speedup:.2f}x")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

results = {
    'fp32': result_fp32,
    'mixed_precision': result_mixed
}

if result_fp32['success'] and result_mixed['success']:
    speedup = result_fp32['mean_ms'] / result_mixed['mean_ms']
    bandwidth_savings = 50.0  # Theoretical: FP16 is 50% of FP32

    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"FP32 time: {result_fp32['mean_ms']:.2f} ms")
    print(f"Mixed precision time: {result_mixed['mean_ms']:.2f} ms")
    print(f"Theoretical bandwidth reduction: {bandwidth_savings:.0f}%")

    if speedup > 1.05:
        print("\n✅ Manual mixed precision provides speedup!")
        print(f"   Recommended for production if accuracy validates")
    elif speedup > 0.95:
        print("\n⚠️  Manual mixed precision shows neutral performance")
        print(f"   May still be useful for memory reduction")
    else:
        print("\n❌ Manual mixed precision is slower")
        print(f"   FP16 conversion overhead outweighs benefits on MPS")

    results['analysis'] = {
        'speedup': float(speedup),
        'recommendation': 'use' if speedup > 1.05 else 'neutral' if speedup > 0.95 else 'avoid'
    }
else:
    print("\n❌ Could not complete analysis due to errors")

# Save results
Path('output').mkdir(exist_ok=True)
with open('output/manual_mixed_precision.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("Results saved to: output/manual_mixed_precision.json")
print("="*70)
