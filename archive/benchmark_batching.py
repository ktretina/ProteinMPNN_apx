#!/usr/bin/env python3
"""
Benchmark batch size optimization for ProteinMPNN on M3 Pro
Tests processing multiple proteins in parallel
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

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

def load_model(model_path):
    """Load model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=128,
        edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=checkpoint['num_edges']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def benchmark_batch(model, pdb_path, batch_size=1, num_runs=20):
    """Benchmark model with different batch sizes."""
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]
    
    # Create batch by replicating the protein
    batch_clones = [copy.deepcopy(protein) for _ in range(batch_size)]
    
    X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
        batch_clones, device, None, None, None, None, None, None, ca_only=False
    )
    
    seq_length = int(mask[0].sum().item())
    
    # Warmup
    print(f"  Warmup (batch_size={batch_size})...")
    with torch.no_grad():
        for _ in range(3):
            randn = torch.randn(chain_M.shape, device=device)
            _ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
            if device.type == 'mps':
                torch.mps.synchronize()
    
    # Timing
    print(f"  Timing (batch_size={batch_size})...")
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'mps':
                torch.mps.synchronize()
            start = time.perf_counter()
            randn = torch.randn(chain_M.shape, device=device)
            _ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
            if device.type == 'mps':
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times)
    mean_time = np.mean(times)
    
    # Calculate throughput
    total_residues = seq_length * batch_size
    throughput = total_residues / mean_time
    time_per_protein = mean_time / batch_size
    
    return {
        'batch_size': batch_size,
        'length': seq_length,
        'total_residues': total_residues,
        'mean_time_ms': float(mean_time * 1000),
        'std_time_ms': float(np.std(times) * 1000),
        'time_per_protein_ms': float(time_per_protein * 1000),
        'throughput_res_per_sec': float(throughput),
        'throughput_proteins_per_sec': float(1.0 / time_per_protein)
    }

pdb_path = '/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb'
model_path = '/Users/ktretina/claude_dir/ProteinMPNN/vanilla_model_weights/v_48_020.pt'

print("\n" + "="*70)
print("BATCH SIZE OPTIMIZATION BENCHMARK")
print("="*70)

model = load_model(model_path)

batch_sizes = [1, 2, 4, 8]
results = {'metadata': {'pdb': pdb_path}, 'batches': {}}

for batch_size in batch_sizes:
    print(f"\n{'-'*70}")
    print(f"BATCH SIZE: {batch_size}")
    print(f"{'-'*70}")
    
    result = benchmark_batch(model, pdb_path, batch_size=batch_size)
    results['batches'][str(batch_size)] = result
    
    print(f"Total time: {result['mean_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
    print(f"Time per protein: {result['time_per_protein_ms']:.2f} ms")
    print(f"Throughput: {result['throughput_res_per_sec']:.1f} res/sec")
    print(f"Proteins/sec: {result['throughput_proteins_per_sec']:.2f}")

# Calculate efficiency
print("\n" + "="*70)
print("BATCH EFFICIENCY ANALYSIS")
print("="*70)

baseline_time_per_protein = results['batches']['1']['time_per_protein_ms']
print(f"\nBaseline (batch=1): {baseline_time_per_protein:.2f} ms/protein")
print(f"\n{'Batch Size':<12} {'Time/Protein':<15} {'Speedup':<12} {'Efficiency':<12}")
print("-" * 51)

for batch_size in batch_sizes:
    result = results['batches'][str(batch_size)]
    time_per_protein = result['time_per_protein_ms']
    speedup = baseline_time_per_protein / time_per_protein
    efficiency = speedup / batch_size * 100  # % of ideal scaling
    
    print(f"{batch_size:<12} {time_per_protein:<15.2f} {speedup:<12.2f}x {efficiency:<12.1f}%")

Path('output').mkdir(exist_ok=True)
with open('output/batching_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: output/batching_benchmarks.json")
