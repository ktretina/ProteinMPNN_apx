#!/usr/bin/env python3
"""
Test model pruning optimization - reducing layers and hidden dimensions
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
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        augment_eps=0.0,
        k_neighbors=k_neighbors
    )
    
    # Load weights where dimensions match, skip mismatched ones
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model

def benchmark_variant(model, pdb_path, num_runs=20):
    """Benchmark a model variant."""
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
print("MODEL PRUNING OPTIMIZATION - Reduced Layers & Dimensions")
print("="*70)

variants = {
    'baseline': {
        'name': 'Baseline (3+3 layers, dim=128)',
        'encoder_layers': 3,
        'decoder_layers': 3,
        'hidden_dim': 128,
        'k_neighbors': 48
    },
    'fewer_layers': {
        'name': 'Fewer Layers (2+2 layers, dim=128)',
        'encoder_layers': 2,
        'decoder_layers': 2,
        'hidden_dim': 128,
        'k_neighbors': 48
    },
    'smaller_dim': {
        'name': 'Smaller Dim (3+3 layers, dim=64)',
        'encoder_layers': 3,
        'decoder_layers': 3,
        'hidden_dim': 64,
        'k_neighbors': 48
    },
    'minimal': {
        'name': 'Minimal (2+2 layers, dim=64)',
        'encoder_layers': 2,
        'decoder_layers': 2,
        'hidden_dim': 64,
        'k_neighbors': 48
    },
    'minimal_fast': {
        'name': 'Minimal+Fast (2+2 layers, dim=64, k=16)',
        'encoder_layers': 2,
        'decoder_layers': 2,
        'hidden_dim': 64,
        'k_neighbors': 16
    }
}

results = {'variants': {}}

for variant_key, config in variants.items():
    print(f"\n{'-'*70}")
    print(f"VARIANT: {config['name']}")
    print(f"{'-'*70}")
    
    try:
        model = load_model(
            model_path,
            num_encoder_layers=config['encoder_layers'],
            num_decoder_layers=config['decoder_layers'],
            hidden_dim=config['hidden_dim'],
            k_neighbors=config['k_neighbors']
        )
        
        result = benchmark_variant(model, pdb_path)
        results['variants'][variant_key] = {
            'name': config['name'],
            'config': config,
            'result': result,
            'status': 'success'
        }
        
        print(f"Time: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
        print(f"Throughput: {result['throughput']:.1f} res/sec")
        
    except Exception as e:
        print(f"Failed: {str(e)}")
        results['variants'][variant_key] = {
            'name': config['name'],
            'config': config,
            'status': 'failed',
            'error': str(e)
        }

# Calculate speedups
baseline_time = results['variants']['baseline']['result']['mean_ms']

print("\n" + "="*70)
print("SPEEDUP ANALYSIS")
print("="*70)
print(f"\n{'Variant':<35} {'Time (ms)':<12} {'Speedup':<10}")
print("-"*57)

for variant_key, data in results['variants'].items():
    if data['status'] == 'success':
        time_ms = data['result']['mean_ms']
        speedup = baseline_time / time_ms
        print(f"{data['name']:<35} {time_ms:<12.2f} {speedup:<10.2f}x")

Path('output').mkdir(exist_ok=True)
with open('output/model_pruning_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: output/model_pruning_benchmarks.json")
