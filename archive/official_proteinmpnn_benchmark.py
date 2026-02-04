#!/usr/bin/env python3
"""
Comprehensive Benchmark of Official ProteinMPNN on M3 Pro Hardware

This script benchmarks the official ProteinMPNN implementation on Apple Silicon M3 Pro,
measuring inference time, throughput, and memory usage across different sequence lengths.
"""

import torch
import numpy as np
import time
import json
import sys
import os
from pathlib import Path
from typing import Dict, List
import argparse

# Import ProteinMPNN modules
from protein_mpnn_utils import ProteinMPNN, StructureDatasetPDB

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
    """
    Create a realistic test protein structure (alpha helix).

    Args:
        length: Number of residues
        output_path: Path to save PDB file

    Returns:
        Path to created PDB file
    """
    if output_path is None:
        output_path = f"test_protein_{length}.pdb"

    # Alpha helix: 3.6 residues/turn, 1.5 Å rise per residue, 5 Å radius
    t = np.linspace(0, length * 2 * np.pi / 3.6, length)

    with open(output_path, 'w') as f:
        f.write("HEADER    TEST PROTEIN\n")
        f.write("TITLE     SYNTHETIC ALPHA HELIX FOR BENCHMARKING\n")

        for i, angle in enumerate(t):
            res_num = i + 1
            # CA coordinates
            ca_x = 5.0 * np.cos(angle)
            ca_y = 5.0 * np.sin(angle)
            ca_z = 1.5 * i

            # Approximate N, C, O positions relative to CA
            n_x = ca_x - 0.5
            n_y = ca_y
            n_z = ca_z - 1.2

            c_x = ca_x + 0.5
            c_y = ca_y
            c_z = ca_z + 1.2

            o_x = ca_x + 1.0
            o_y = ca_y
            o_z = ca_z + 1.8

            # Write atoms (using ALA as placeholder)
            f.write(f"ATOM  {i*4+1:5d}  N   ALA A{res_num:4d}    {n_x:8.3f}{n_y:8.3f}{n_z:8.3f}  1.00  0.00           N\n")
            f.write(f"ATOM  {i*4+2:5d}  CA  ALA A{res_num:4d}    {ca_x:8.3f}{ca_y:8.3f}{ca_z:8.3f}  1.00  0.00           C\n")
            f.write(f"ATOM  {i*4+3:5d}  C   ALA A{res_num:4d}    {c_x:8.3f}{c_y:8.3f}{c_z:8.3f}  1.00  0.00           C\n")
            f.write(f"ATOM  {i*4+4:5d}  O   ALA A{res_num:4d}    {o_x:8.3f}{o_y:8.3f}{o_z:8.3f}  1.00  0.00           O\n")

        f.write("END\n")

    return output_path


def load_model(model_path: str, device: torch.device, ca_only: bool = False):
    """Load ProteinMPNN model checkpoint following official pattern."""
    checkpoint = torch.load(model_path, map_location=device)

    hidden_dim = 128
    num_layers = 3

    print(f"\nModel loaded from: {model_path}")
    print(f"  Number of edges (k-neighbors): {checkpoint['num_edges']}")
    print(f"  Training noise level: {checkpoint.get('noise_level', 'N/A')}Å")

    model = ProteinMPNN(
        ca_only=ca_only,
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
    model.to(device)
    model.eval()

    return model, checkpoint


def benchmark_inference(model, pdb_path: str, num_runs: int = 50,
                       warmup_runs: int = 10, device: torch.device = None,
                       ca_only: bool = False) -> Dict:
    """
    Benchmark inference time for a single protein using official workflow.

    Args:
        model: ProteinMPNN model
        pdb_path: Path to PDB file
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs
        device: Device to use
        ca_only: Whether using CA-only model

    Returns:
        Dictionary with timing statistics
    """
    # Use official workflow: parse_PDB -> tied_featurize
    from protein_mpnn_utils import parse_PDB, tied_featurize
    import copy

    pdb_dict_list = parse_PDB(pdb_path, ca_only=ca_only)
    protein = pdb_dict_list[0]

    # Create batch (following official pattern)
    batch_clones = [copy.deepcopy(protein)]

    # Featurize using official function
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch_clones, device, None, None, None, None, None, None, ca_only=ca_only
    )

    seq_length = int(mask.sum().item())

    # Warmup runs
    print(f"  Running {warmup_runs} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            randn = torch.randn(chain_M.shape, device=device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

    # Actual timing
    print(f"  Running {num_runs} timed iterations...")
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            randn = torch.randn(chain_M.shape, device=device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)

            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    min_time = np.min(times)

    throughput = seq_length / mean_time

    return {
        'sequence_length': int(seq_length),
        'mean_time_sec': float(mean_time),
        'std_time_sec': float(std_time),
        'median_time_sec': float(median_time),
        'min_time_sec': float(min_time),
        'mean_time_ms': float(mean_time * 1000),
        'throughput_res_per_sec': float(throughput),
        'num_runs': num_runs,
        'warmup_runs': warmup_runs
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark ProteinMPNN on M3 Pro')
    parser.add_argument('--model_path', type=str,
                       default='vanilla_model_weights/v_48_020.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--lengths', type=int, nargs='+',
                       default=[50, 100, 200, 500],
                       help='Sequence lengths to test')
    parser.add_argument('--num_runs', type=int, default=50,
                       help='Number of timing runs per test')
    parser.add_argument('--warmup_runs', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--ca_only', action='store_true',
                       help='Use CA-only model')

    args = parser.parse_args()

    print("="*70)
    print("OFFICIAL PROTEINMPNN BENCHMARK - M3 Pro Hardware")
    print("="*70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Test Lengths: {args.lengths}")
    print(f"Runs per test: {args.num_runs} (+ {args.warmup_runs} warmup)")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(args.model_path, device, args.ca_only)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for test PDBs
    temp_dir = Path("temp_benchmark_pdbs")
    temp_dir.mkdir(exist_ok=True)

    # Run benchmarks
    results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'pytorch_version': torch.__version__,
            'device': str(device),
            'model_path': args.model_path,
            'test_lengths': args.lengths,
            'num_runs': args.num_runs,
            'warmup_runs': args.warmup_runs,
            'num_edges': checkpoint['num_edges'],
            'noise_level': checkpoint.get('noise_level', 'N/A')
        },
        'benchmarks': {}
    }

    for length in args.lengths:
        print(f"\n{'='*70}")
        print(f"Benchmarking {length}-residue protein")
        print(f"{'='*70}")

        # Create test protein
        pdb_path = temp_dir / f"test_{length}.pdb"
        create_test_protein(length, str(pdb_path))

        # Benchmark
        result = benchmark_inference(
            model, str(pdb_path),
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            device=device,
            ca_only=args.ca_only
        )

        results['benchmarks'][str(length)] = result

        print(f"\nResults:")
        print(f"  Mean time: {result['mean_time_ms']:.2f} ± {result['std_time_sec']*1000:.2f} ms")
        print(f"  Median time: {result['median_time_sec']*1000:.2f} ms")
        print(f"  Min time: {result['min_time_sec']*1000:.2f} ms")
        print(f"  Throughput: {result['throughput_res_per_sec']:.1f} res/sec")

    # Save results
    output_file = output_dir / 'official_proteinmpnn_benchmarks.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Summary table
    print("\nSUMMARY:")
    print(f"{'Length':<10} {'Mean (ms)':<12} {'Throughput (res/s)':<20}")
    print("-" * 42)
    for length in args.lengths:
        r = results['benchmarks'][str(length)]
        print(f"{length:<10} {r['mean_time_ms']:<12.2f} {r['throughput_res_per_sec']:<20.1f}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    return results


if __name__ == '__main__':
    results = main()
