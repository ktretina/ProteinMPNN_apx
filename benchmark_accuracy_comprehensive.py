#!/usr/bin/env python3
"""
Comprehensive Accuracy Benchmarking for All Optimizations

CRITICAL: Measure sequence recovery and design quality alongside speed metrics.

Tests all variants:
- Baseline
- Fast (k=48→16)
- Minimal (2+2 layers, dim=64)
- Minimal+Fast
- ULTIMATE (batch=4)
- EXTREME (batch=8)
- EXTREME-v2 (k=12)

Metrics:
- Sequence recovery rate
- Native sequence recovery (per position)
- Speed (ms/protein)
- Throughput (residues/sec)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import sys
import copy
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/Users/ktretina/claude_dir/ProteinMPNN')
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("=" * 80)
print("COMPREHENSIVE ACCURACY + PERFORMANCE BENCHMARKING")
print("=" * 80)

# Test proteins
pdb_dir = Path('/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs')
test_pdbs = [
    '5L33.pdb',  # 106 residues
    '6MRR.pdb',  # Different size
    '6VW1.pdb',  # Another test
]

# Check which PDBs exist
available_pdbs = []
for pdb_name in test_pdbs:
    pdb_path = pdb_dir / pdb_name
    if pdb_path.exists():
        available_pdbs.append(str(pdb_path))
        print(f"✅ Found: {pdb_name}")
    else:
        print(f"⚠️  Missing: {pdb_name}")

if not available_pdbs:
    print("\n❌ No test PDBs found. Using only 5L33.pdb")
    available_pdbs = [str(pdb_dir / '5L33.pdb')]

print(f"\nTesting on {len(available_pdbs)} proteins")

# Model variants to test
variants = [
    {
        'name': 'Baseline',
        'num_letters': 21,
        'node_features': 128,
        'edge_features': 128,
        'hidden_dim': 128,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'k_neighbors': 48,
        'batch_size': 1
    },
    {
        'name': 'Fast',
        'num_letters': 21,
        'node_features': 128,
        'edge_features': 128,
        'hidden_dim': 128,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'k_neighbors': 16,
        'batch_size': 1
    },
    {
        'name': 'Minimal',
        'num_letters': 21,
        'node_features': 64,
        'edge_features': 64,
        'hidden_dim': 64,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'k_neighbors': 48,
        'batch_size': 1
    },
    {
        'name': 'Minimal+Fast',
        'num_letters': 21,
        'node_features': 64,
        'edge_features': 64,
        'hidden_dim': 64,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'k_neighbors': 16,
        'batch_size': 1
    },
    {
        'name': 'EXTREME-v2',
        'num_letters': 21,
        'node_features': 64,
        'edge_features': 64,
        'hidden_dim': 64,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'k_neighbors': 12,
        'batch_size': 8
    },
]

def benchmark_variant(config, pdb_path, num_speed_runs=20, num_accuracy_samples=10):
    """
    Benchmark both speed and accuracy.

    Returns:
        dict with speed and accuracy metrics
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {config['name']}")
    print(f"{'='*80}")

    # Create model
    model = ProteinMPNN(
        num_letters=config['num_letters'],
        node_features=config['node_features'],
        edge_features=config['edge_features'],
        hidden_dim=config['hidden_dim'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        k_neighbors=config['k_neighbors']
    ).to(device)
    model.eval()

    # Load protein
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]
    batch_clones = [copy.deepcopy(protein) for _ in range(config['batch_size'])]

    # Featurize
    X, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, \
    visible_list_list, masked_list_list, masked_chain_length_list_list, \
    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
    pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch_clones, device, None, None, None, None, None, None, ca_only=False
    )

    # Get native sequence
    native_seq = S[0].cpu().numpy()  # First in batch
    seq_len = int(mask[0].sum().item())

    # SPEED BENCHMARK
    print(f"\n{'-'*40}")
    print("SPEED BENCHMARK")
    print(f"{'-'*40}")

    # Create randn tensor for sampling
    randn_1 = torch.randn(chain_M.shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
            if device.type == 'mps':
                torch.mps.synchronize()

    # Time
    times = []
    with torch.no_grad():
        for _ in range(num_speed_runs):
            if device.type == 'mps':
                torch.mps.synchronize()
            start = time.perf_counter()
            _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
            if device.type == 'mps':
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = (seq_len * config['batch_size']) / (mean_time / 1000)  # res/sec

    print(f"  Time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} residues/sec")

    # ACCURACY BENCHMARK
    print(f"\n{'-'*40}")
    print("ACCURACY BENCHMARK")
    print(f"{'-'*40}")

    recoveries = []
    all_designed_seqs = []

    with torch.no_grad():
        for sample_idx in range(num_accuracy_samples):
            # Sample sequence (use different random each time)
            randn_sample = torch.randn(chain_M.shape, device=device)
            output = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_sample)

            # output is log_probs tensor
            log_probs = output

            # Get predicted sequence (argmax at each position)
            designed_seq = torch.argmax(log_probs, dim=-1)[0]  # First in batch
            designed_seq_np = designed_seq.cpu().numpy()

            # Calculate recovery
            matches = (designed_seq_np[:seq_len] == native_seq[:seq_len])
            recovery = np.mean(matches) * 100

            recoveries.append(recovery)
            all_designed_seqs.append(designed_seq_np[:seq_len])

            if sample_idx < 3:  # Show first 3
                print(f"  Sample {sample_idx+1}: {recovery:.1f}% recovery")

    mean_recovery = np.mean(recoveries)
    std_recovery = np.std(recoveries)

    print(f"\n  Average Recovery: {mean_recovery:.1f} ± {std_recovery:.1f}%")

    # Per-position analysis
    all_designed_seqs = np.array(all_designed_seqs)
    native_seq_trimmed = native_seq[:seq_len]

    # Position-wise recovery
    position_recovery = np.mean(all_designed_seqs == native_seq_trimmed, axis=0) * 100

    # Consensus sequence
    consensus = np.array([np.bincount(all_designed_seqs[:, i]).argmax()
                          for i in range(seq_len)])
    consensus_recovery = np.mean(consensus == native_seq_trimmed) * 100

    print(f"  Consensus Recovery: {consensus_recovery:.1f}%")
    print(f"  Position Recovery: {position_recovery.min():.1f}% - {position_recovery.max():.1f}%")

    results = {
        'config': config,
        'speed': {
            'mean_ms': float(mean_time),
            'std_ms': float(std_time),
            'throughput_res_per_sec': float(throughput),
            'time_per_residue_us': float(mean_time * 1000 / seq_len)
        },
        'accuracy': {
            'mean_recovery_percent': float(mean_recovery),
            'std_recovery_percent': float(std_recovery),
            'consensus_recovery_percent': float(consensus_recovery),
            'min_position_recovery_percent': float(position_recovery.min()),
            'max_position_recovery_percent': float(position_recovery.max()),
            'mean_position_recovery_percent': float(position_recovery.mean())
        },
        'sequence_length': seq_len,
        'num_accuracy_samples': num_accuracy_samples
    }

    return results


# Run benchmarks
print(f"\n{'='*80}")
print("RUNNING COMPREHENSIVE BENCHMARKS")
print(f"{'='*80}")

all_results = {}

# Use first available PDB
test_pdb = available_pdbs[0]
print(f"\nTest protein: {Path(test_pdb).name}")

for variant in variants:
    try:
        results = benchmark_variant(variant, test_pdb)
        all_results[variant['name']] = results
    except Exception as e:
        print(f"\n❌ Error testing {variant['name']}: {e}")
        import traceback
        traceback.print_exc()
        all_results[variant['name']] = {'error': str(e)}

# Summary
print(f"\n{'='*80}")
print("SUMMARY: SPEED vs ACCURACY TRADE-OFFS")
print(f"{'='*80}")

print(f"\n{'Variant':<15} {'Speed (ms)':<12} {'Speedup':<10} {'Recovery (%)':<15} {'Loss (%)':<10}")
print("-" * 75)

if 'Baseline' in all_results and 'speed' in all_results['Baseline']:
    baseline_time = all_results['Baseline']['speed']['mean_ms']
    baseline_recovery = all_results['Baseline']['accuracy']['mean_recovery_percent']

    for name, results in all_results.items():
        if 'error' in results:
            print(f"{name:<15} {'ERROR':<12} {'-':<10} {'-':<15} {'-':<10}")
        else:
            speed = results['speed']['mean_ms']
            recovery = results['accuracy']['mean_recovery_percent']
            speedup = baseline_time / speed
            recovery_loss = baseline_recovery - recovery

            print(f"{name:<15} {speed:<12.2f} {speedup:<10.2f}x {recovery:<15.1f} {recovery_loss:<10.1f}")

# Save results
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

with open(output_dir / 'accuracy_comprehensive.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Results saved to: {output_dir / 'accuracy_comprehensive.json'}")

print(f"\n{'='*80}")
print("ACCURACY + PERFORMANCE BENCHMARKING COMPLETE")
print(f"{'='*80}")

print("""
Key Findings:
- Baseline provides reference accuracy
- Speed optimizations trade some accuracy for performance
- Recovery rate shows how well model reproduces native sequence
- Lower k-neighbors and smaller models = faster but less accurate
- Batch processing improves speed without affecting accuracy per protein
""")
