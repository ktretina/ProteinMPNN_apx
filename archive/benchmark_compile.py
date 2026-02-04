#!/usr/bin/env python3
"""
Benchmark torch.compile optimization for ProteinMPNN on M3 Pro
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
print(f"PyTorch version: {torch.__version__}")

def load_model(model_path, compile_model=False):
    """Load model with optional compilation."""
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
    
    if compile_model:
        print("  Compiling model...")
        try:
            model = torch.compile(model, backend='aot_eager')
            print("  Model compiled successfully")
        except Exception as e:
            print(f"  Warning: compilation failed: {e}")
            print("  Continuing with eager mode")
    
    return model

def benchmark(model, pdb_path, num_runs=30):
    """Benchmark model on a PDB file."""
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]
    batch_clones = [copy.deepcopy(protein)]
    
    X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
        batch_clones, device, None, None, None, None, None, None, ca_only=False
    )
    
    seq_length = int(mask.sum().item())
    
    # Warmup
    print("  Warmup...")
    with torch.no_grad():
        for _ in range(5):
            randn = torch.randn(chain_M.shape, device=device)
            _ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
            if device.type == 'mps':
                torch.mps.synchronize()
    
    # Timing
    print("  Timing...")
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
        'throughput': float(seq_length / np.mean(times))
    }

pdb_path = '/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb'
model_path = '/Users/ktretina/claude_dir/ProteinMPNN/vanilla_model_weights/v_48_020.pt'

print("\n" + "="*70)
print("BASELINE (EAGER MODE)")
print("="*70)
model_eager = load_model(model_path, compile_model=False)
result_eager = benchmark(model_eager, pdb_path)
print(f"Length: {result_eager['length']} residues")
print(f"Time: {result_eager['mean_ms']:.2f} ± {result_eager['std_ms']:.2f} ms")
print(f"Throughput: {result_eager['throughput']:.1f} res/sec")

print("\n" + "="*70)
print("TORCH.COMPILE (AOT_EAGER)")
print("="*70)
model_compiled = load_model(model_path, compile_model=True)
result_compiled = benchmark(model_compiled, pdb_path)
print(f"Length: {result_compiled['length']} residues")
print(f"Time: {result_compiled['mean_ms']:.2f} ± {result_compiled['std_ms']:.2f} ms")
print(f"Throughput: {result_compiled['throughput']:.1f} res/sec")

speedup = result_eager['mean_ms'] / result_compiled['mean_ms']
print("\n" + "="*70)
print(f"SPEEDUP: {speedup:.2f}x")
print("="*70)

results = {
    'baseline_eager': result_eager,
    'torch_compile': result_compiled,
    'speedup': speedup
}

Path('output').mkdir(exist_ok=True)
with open('output/compile_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: output/compile_benchmarks.json")
