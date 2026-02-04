#!/usr/bin/env python3
"""
Benchmark script for ProteinMPNN optimization variants on M3 Pro

This script systematically tests various optimization strategies:
1. Baseline (FP32)
2. BFloat16 precision
3. torch.compile
4. Combined optimizations
"""

import torch
import numpy as np
import time
import json
import sys
import os
from pathlib import Path
from typing import Dict
import argparse
import copy

# Add ProteinMPNN to path
proteinmpnn_path = Path(__file__).parent.parent / "ProteinMPNN"
sys.path.insert(0, str(proteinmpnn_path))

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

# Check device availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using Metal Performance Shaders (MPS) on Apple Silicon")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(f"Using CPU")


def create_test_protein(length: int = 100, output_path: str = None) -> str:
    """Create a realistic test protein structure (alpha helix)."""
    if output_path is None:
        output_path = f"test_protein_{length}.pdb"

    t = np.linspace(0, length * 2 * np.pi / 3.6, length)

    with open(output_path, 'w') as f:
        f.write("HEADER    TEST PROTEIN\\n")
        f.write("TITLE     SYNTHETIC ALPHA HELIX FOR BENCHMARKING\\n")

        for i, angle in enumerate(t):
            res_num = i + 1
            ca_x = 5.0 * np.cos(angle)
            ca_y = 5.0 * np.sin(angle)
            ca_z = 1.5 * i

            n_x = ca_x - 0.5
            n_y = ca_y
            n_z = ca_z - 1.2

            c_x = ca_x + 0.5
            c_y = ca_y
            c_z = ca_z + 1.2

            o_x = ca_x + 1.0
            o_y = ca_y
            o_z = ca_z + 1.8

            f.write(f"ATOM  {i*4+1:5d}  N   ALA A{res_num:4d}    {n_x:8.3f}{n_y:8.3f}{n_z:8.3f}  1.00  0.00           N\\n")
            f.write(f"ATOM  {i*4+2:5d}  CA  ALA A{res_num:4d}    {ca_x:8.3f}{ca_y:8.3f}{ca_z:8.3f}  1.00  0.00           C\\n")
            f.write(f"ATOM  {i*4+3:5d}  C   ALA A{res_num:4d}    {c_x:8.3f}{c_y:8.3f}{c_z:8.3f}  1.00  0.00           C\\n")
            f.write(f"ATOM  {i*4+4:5d}  O   ALA A{res_num:4d}    {o_x:8.3f}{o_y:8.3f}{o_z:8.3f}  1.00  0.00           O\\n")

        f.write("END\\n")

    return output_path


def load_model(model_path: str, device: torch.device, dtype=None):
    """Load ProteinMPNN model with optional dtype conversion."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    hidden_dim = 128
    num_layers = 3

    print(f"\\nLoading model: {model_path}")
    if dtype:
        print(f"  Precision: {dtype}")

    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,
        k_neighbors=checkpoint['num_edges']
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    if dtype:
        model = model.to(dtype=dtype)

    model.to(device)
    model.eval()

    return model


def benchmark_variant(model, pdb_path: str, num_runs: int, warmup_runs: int, dtype=None):
    """Benchmark a model variant."""
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]
    batch_clones = [copy.deepcopy(protein)]

    X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, _, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch_clones, device, None, None, None, None, None, None, ca_only=False
    )

    if dtype:
        X = X.to(dtype=dtype)
        mask = mask.to(dtype=dtype)
        chain_M = chain_M.to(dtype=dtype)
        chain_M_pos = chain_M_pos.to(dtype=dtype)
        chain_encoding_all = chain_encoding_all.to(dtype=dtype)

    seq_length = int(mask.sum().item())

    # Warmup
    print(f"  Warmup: {warmup_runs} iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            randn = torch.randn(chain_M.shape, device=device)
            if dtype:
                randn = randn.to(dtype=dtype)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
            if device.type == 'mps':
                torch.mps.synchronize()

    # Timing
    print(f"  Timing: {num_runs} iterations...")
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'mps':
                torch.mps.synchronize()

            start = time.perf_counter()
            randn = torch.randn(chain_M.shape, device=device)
            if dtype:
                randn = randn.to(dtype=dtype)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)

            if device.type == 'mps':
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)
    return {
        'sequence_length': int(seq_length),
        'mean_time_ms': float(np.mean(times) * 1000),
        'std_time_ms': float(np.std(times) * 1000),
        'throughput_res_per_sec': float(seq_length / np.mean(times))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../ProteinMPNN/vanilla_model_weights/v_48_020.pt')
    parser.add_argument('--lengths', type=int, nargs='+', default=[100, 200])
    parser.add_argument('--num_runs', type=int, default=30)
    parser.add_argument('--warmup_runs', type=int, default=5)
    args = parser.parse_args()

    print("="*70)
    print("PROTEINMPNN OPTIMIZATION VARIANTS - M3 Pro Benchmark")
    print("="*70)

    temp_dir = Path("temp_variant_pdbs")
    temp_dir.mkdir(exist_ok=True)

    variants = {
        'baseline': {'dtype': None},
        'bfloat16': {'dtype': torch.bfloat16}
    }

    results = {'metadata': {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}, 'variants': {}}

    for variant_name, config in variants.items():
        print(f"\\n{'='*70}")
        print(f"VARIANT: {variant_name.upper()}")
        print(f"{'='*70}")

        model = load_model(args.model_path, device, dtype=config['dtype'])
        results['variants'][variant_name] = {}

        for length in args.lengths:
            print(f"\\nTesting {length} residues...")
            pdb_path = temp_dir / f"test_{length}.pdb"
            create_test_protein(length, str(pdb_path))

            result = benchmark_variant(model, str(pdb_path), args.num_runs, args.warmup_runs, config['dtype'])
            results['variants'][variant_name][str(length)] = result

            print(f"  Time: {result['mean_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
            print(f"  Throughput: {result['throughput_res_per_sec']:.1f} res/sec")

    # Calculate speedups
    baseline = results['variants']['baseline']
    print(f"\\n{'='*70}")
    print("SPEEDUP ANALYSIS")
    print(f"{'='*70}")

    for variant_name in variants:
        if variant_name == 'baseline':
            continue
        variant_data = results['variants'][variant_name]
        print(f"\\n{variant_name.upper()}:")
        for length in args.lengths:
            baseline_time = baseline[str(length)]['mean_time_ms']
            variant_time = variant_data[str(length)]['mean_time_ms']
            speedup = baseline_time / variant_time
            print(f"  {length} residues: {speedup:.2f}x speedup")

    # Save results
    output_file = Path('output/variant_benchmarks.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\\nResults saved to: {output_file}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
