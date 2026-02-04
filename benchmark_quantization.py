#!/usr/bin/env python3
"""
Test Int8 Quantization optimization for ProteinMPNN
Tests dynamic quantization to reduce memory bandwidth and improve performance
"""

import torch
import numpy as np
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/ktretina/claude_dir/ProteinMPNN')

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

def load_model(model_path, device):
    """Load baseline model."""
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

def benchmark_model(model, pdb_path, device, num_runs=20):
    """Benchmark a model."""
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
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            randn = torch.randn(chain_M.shape, device=device)
            _ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)

            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

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
print("INT8 QUANTIZATION OPTIMIZATION TEST")
print("="*70)

# Test on MPS first
print("\n" + "-"*70)
print("Testing on MPS (Apple Silicon)")
print("-"*70)

try:
    device_mps = torch.device("mps")
    model_mps = load_model(model_path, device_mps)

    print("\n1. BASELINE (FP32 on MPS)")
    result_baseline_mps = benchmark_model(model_mps, pdb_path, device_mps)
    print(f"Time: {result_baseline_mps['mean_ms']:.2f} ± {result_baseline_mps['std_ms']:.2f} ms")
    print(f"Throughput: {result_baseline_mps['throughput']:.1f} res/sec")

    # Try dynamic quantization on MPS
    print("\n2. DYNAMIC INT8 QUANTIZATION (on MPS)")
    try:
        model_mps_quant = load_model(model_path, device_mps)

        # Apply dynamic quantization to Linear layers
        model_mps_quant = torch.quantization.quantize_dynamic(
            model_mps_quant,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        result_quant_mps = benchmark_model(model_mps_quant, pdb_path, device_mps)
        print(f"Time: {result_quant_mps['mean_ms']:.2f} ± {result_quant_mps['std_ms']:.2f} ms")
        print(f"Throughput: {result_quant_mps['throughput']:.1f} res/sec")
        print(f"Speedup: {result_baseline_mps['mean_ms'] / result_quant_mps['mean_ms']:.2f}x")
        mps_quant_works = True
    except Exception as e:
        print(f"ERROR: Dynamic quantization failed on MPS: {e}")
        mps_quant_works = False

except Exception as e:
    print(f"ERROR testing on MPS: {e}")
    mps_quant_works = False

# Test on CPU for comparison
print("\n" + "-"*70)
print("Testing on CPU (for comparison)")
print("-"*70)

device_cpu = torch.device("cpu")
model_cpu = load_model(model_path, device_cpu)

print("\n3. BASELINE (FP32 on CPU)")
result_baseline_cpu = benchmark_model(model_cpu, pdb_path, device_cpu, num_runs=10)
print(f"Time: {result_baseline_cpu['mean_ms']:.2f} ± {result_baseline_cpu['std_ms']:.2f} ms")
print(f"Throughput: {result_baseline_cpu['throughput']:.1f} res/sec")

print("\n4. DYNAMIC INT8 QUANTIZATION (on CPU)")
try:
    model_cpu_quant = load_model(model_path, device_cpu)

    # Apply dynamic quantization
    model_cpu_quant = torch.quantization.quantize_dynamic(
        model_cpu_quant,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    result_quant_cpu = benchmark_model(model_cpu_quant, pdb_path, device_cpu, num_runs=10)
    print(f"Time: {result_quant_cpu['mean_ms']:.2f} ± {result_quant_cpu['std_ms']:.2f} ms")
    print(f"Throughput: {result_quant_cpu['throughput']:.1f} res/sec")
    print(f"Speedup: {result_baseline_cpu['mean_ms'] / result_quant_cpu['mean_ms']:.2f}x")
    cpu_quant_speedup = result_baseline_cpu['mean_ms'] / result_quant_cpu['mean_ms']
except Exception as e:
    print(f"ERROR: Dynamic quantization failed on CPU: {e}")
    cpu_quant_speedup = None

# Summary
print("\n" + "="*70)
print("QUANTIZATION SUMMARY")
print("="*70)

results = {
    'mps': {
        'baseline': result_baseline_mps if 'result_baseline_mps' in locals() else None,
        'quantized': result_quant_mps if mps_quant_works else None,
        'works': mps_quant_works
    },
    'cpu': {
        'baseline': result_baseline_cpu,
        'quantized': result_quant_cpu if 'result_quant_cpu' in locals() else None,
        'speedup': cpu_quant_speedup
    }
}

if mps_quant_works:
    mps_speedup = result_baseline_mps['mean_ms'] / result_quant_mps['mean_ms']
    print(f"\n✅ MPS Quantization: {mps_speedup:.2f}x speedup")
else:
    print(f"\n❌ MPS Quantization: Not supported or failed")

if cpu_quant_speedup:
    print(f"✅ CPU Quantization: {cpu_quant_speedup:.2f}x speedup")
else:
    print(f"❌ CPU Quantization: Failed")

print("\nNote: Dynamic quantization works best on CPU with Intel MKL-DNN")
print("MPS backend may not support quantized operations efficiently")

Path('output').mkdir(exist_ok=True)
with open('output/quantization_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: output/quantization_benchmarks.json")
