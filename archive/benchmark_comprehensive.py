#!/usr/bin/env python3
"""
Comprehensive benchmark suite for ProteinMPNN on M3 Pro
Tests multiple proteins of different lengths and configurations
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

def load_model(model_path, k_neighbors=None):
    """Load model with optional k_neighbors override."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if k_neighbors is None:
        k_neighbors = checkpoint['num_edges']
    
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

def benchmark_protein(model, pdb_path, num_runs=20):
    """Benchmark model on a PDB file."""
    try:
        pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
        protein = pdb_dict_list[0]
        batch_clones = [copy.deepcopy(protein)]
        
        X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
            batch_clones, device, None, None, None, None, None, None, ca_only=False
        )
        
        seq_length = int(mask.sum().item())
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                randn = torch.randn(chain_M.shape, device=device)
                _ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
                if device.type == 'mps':
                    torch.mps.synchronize()
        
        # Timing
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
        return {
            'length': seq_length,
            'mean_ms': float(np.mean(times) * 1000),
            'std_ms': float(np.std(times) * 1000),
            'throughput': float(seq_length / np.mean(times)),
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }

# Find all available PDB files
pdb_dir = Path('/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs')
pdb_files = sorted(pdb_dir.glob('*.pdb'))[:10]  # Test first 10

model_path = '/Users/ktretina/claude_dir/ProteinMPNN/vanilla_model_weights/v_48_020.pt'

print("\n" + "="*70)
print("COMPREHENSIVE BENCHMARK - Multiple Proteins")
print("="*70)

# Test 1: Multiple proteins with default settings
print("\nTest 1: Multiple proteins (default k=48)")
print("-"*70)

model = load_model(model_path)
results = {'test1_multiple_proteins': {}}

for pdb_file in pdb_files[:5]:
    protein_name = pdb_file.stem
    print(f"Testing {protein_name}...")
    result = benchmark_protein(model, str(pdb_file))
    if result['status'] == 'success':
        print(f"  Length: {result['length']} res, Time: {result['mean_ms']:.2f} ms, Throughput: {result['throughput']:.1f} res/sec")
        results['test1_multiple_proteins'][protein_name] = result
    else:
        print(f"  Failed: {result['error']}")

# Test 2: Different k-neighbor values
print("\n" + "="*70)
print("Test 2: Different k-neighbor values")
print("-"*70)

test_pdb = pdb_files[0]
k_values = [16, 32, 48]
results['test2_k_neighbors'] = {}

for k in k_values:
    print(f"\nTesting k={k} neighbors...")
    model_k = load_model(model_path, k_neighbors=k)
    result = benchmark_protein(model_k, str(test_pdb))
    if result['status'] == 'success':
        print(f"  Time: {result['mean_ms']:.2f} ms, Throughput: {result['throughput']:.1f} res/sec")
        results['test2_k_neighbors'][f'k{k}'] = result

# Test 3: Sequence length scaling
print("\n" + "="*70)
print("Test 3: Sequence length analysis")
print("-"*70)

results['test3_length_scaling'] = {}
for pdb_file in pdb_files:
    result = benchmark_protein(model, str(pdb_file))
    if result['status'] == 'success':
        length = result['length']
        if length not in results['test3_length_scaling']:
            results['test3_length_scaling'][str(length)] = []
        results['test3_length_scaling'][str(length)].append(result)

# Analyze scaling
print("\nLength scaling analysis:")
print(f"{'Length':<10} {'Count':<8} {'Avg Time (ms)':<15} {'Avg Throughput':<15}")
print("-"*48)
for length_str in sorted(results['test3_length_scaling'].keys(), key=int):
    data = results['test3_length_scaling'][length_str]
    avg_time = np.mean([d['mean_ms'] for d in data])
    avg_throughput = np.mean([d['throughput'] for d in data])
    print(f"{length_str:<10} {len(data):<8} {avg_time:<15.2f} {avg_throughput:<15.1f}")

# Save results
Path('output').mkdir(exist_ok=True)
with open('output/comprehensive_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("Results saved to: output/comprehensive_benchmarks.json")
print("="*70)
