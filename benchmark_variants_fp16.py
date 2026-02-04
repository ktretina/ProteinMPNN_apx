#!/usr/bin/env python3
"""
Simple benchmark comparing baseline vs FP16 for ProteinMPNN
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

def load_model(model_path, dtype=None):
    """Load model with optional dtype."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=128,
        edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=checkpoint['num_edges']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if dtype:
        model = model.to(dtype=dtype)
    model.to(device)
    model.eval()
    
    return model

def benchmark(model, pdb_path, num_runs=30, dtype=None):
    """Benchmark model on a PDB file."""
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]
    batch_clones = [copy.deepcopy(protein)]
    
    X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
        batch_clones, device, None, None, None, None, None, None, ca_only=False
    )
    
    # Only convert float tensors, not integer indices
    if dtype == torch.float16:
        X = X.half()
        mask = mask.half()
        chain_M = chain_M.half()
        chain_M_pos = chain_M_pos.half()
        chain_encoding_all = chain_encoding_all.half()
    
    seq_length = int(mask.sum().item())
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            randn = torch.randn(chain_M.shape, device=device)
            if dtype == torch.float16:
                randn = randn.half()
            _ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
            torch.mps.synchronize()
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.mps.synchronize()
            start = time.perf_counter()
            randn = torch.randn(chain_M.shape, device=device)
            if dtype == torch.float16:
                randn = randn.half()
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

print("\n" + "="*70)
print("BASELINE (FP32)")
print("="*70)
model_fp32 = load_model(model_path)
result_fp32 = benchmark(model_fp32, pdb_path)
print(f"Length: {result_fp32['length']} residues")
print(f"Time: {result_fp32['mean_ms']:.2f} ± {result_fp32['std_ms']:.2f} ms")
print(f"Throughput: {result_fp32['throughput']:.1f} res/sec")

print("\n" + "="*70)
print("FP16 (HALF PRECISION)")
print("="*70)
model_fp16 = load_model(model_path, dtype=torch.float16)
result_fp16 = benchmark(model_fp16, pdb_path, dtype=torch.float16)
print(f"Length: {result_fp16['length']} residues")
print(f"Time: {result_fp16['mean_ms']:.2f} ± {result_fp16['std_ms']:.2f} ms")
print(f"Throughput: {result_fp16['throughput']:.1f} res/sec")

speedup = result_fp32['mean_ms'] / result_fp16['mean_ms']
print("\n" + "="*70)
print(f"SPEEDUP: {speedup:.2f}x")
print("="*70)

results = {
    'baseline_fp32': result_fp32,
    'fp16': result_fp16,
    'speedup': speedup
}

Path('output').mkdir(exist_ok=True)
with open('output/precision_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: output/precision_benchmarks.json")
