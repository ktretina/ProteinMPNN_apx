#!/usr/bin/env python3
"""
Run All Model Variants and Save Actual Outputs

This script runs each optimization variant and saves:
- Designed sequences (FASTA format)
- Native sequence for comparison
- Recovery metrics
- Log probabilities
- Detailed comparison
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import copy
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/Users/ktretina/claude_dir/ProteinMPNN')
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("=" * 80)
print("RUNNING ALL VARIANTS WITH SEQUENCE OUTPUTS")
print("=" * 80)

# Create output directory
output_dir = Path('output/model_outputs')
output_dir.mkdir(parents=True, exist_ok=True)

# Test protein
pdb_path = '/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb'

# Amino acid alphabet
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

# Model variants
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
        'batch_size': 1,
        'description': '3+3 layers, dim=128, k=48 (reference)'
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
        'batch_size': 1,
        'description': '3+3 layers, dim=128, k=16 (reduced neighbors)'
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
        'batch_size': 1,
        'description': '2+2 layers, dim=64, k=48 (pruned model)'
    },
    {
        'name': 'Minimal_Fast',
        'num_letters': 21,
        'node_features': 64,
        'edge_features': 64,
        'hidden_dim': 64,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'k_neighbors': 16,
        'batch_size': 1,
        'description': '2+2 layers, dim=64, k=16 (pruned + reduced)'
    },
    {
        'name': 'EXTREME_v2',
        'num_letters': 21,
        'node_features': 64,
        'edge_features': 64,
        'hidden_dim': 64,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'k_neighbors': 12,
        'batch_size': 1,  # Use batch=1 for sequence generation
        'description': '2+2 layers, dim=64, k=12 (maximum speed)'
    },
]

def sequences_to_indices(sequences):
    """Convert AA sequences to integer indices."""
    aa_to_idx = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}
    return [[aa_to_idx.get(aa, 20) for aa in seq] for seq in sequences]

def indices_to_sequences(indices):
    """Convert integer indices to AA sequences."""
    return [''.join([AA_ALPHABET[idx] if idx < 21 else 'X' for idx in seq]) for seq in indices]

def run_variant(config, pdb_path, num_samples=5, temperature=0.1):
    """
    Run a variant and generate sequences.

    Returns dict with sequences and metrics.
    """
    print(f"\n{'='*80}")
    print(f"VARIANT: {config['name']}")
    print(f"{'='*80}")
    print(f"Description: {config['description']}")

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
    batch_clones = [copy.deepcopy(protein) for _ in range(1)]

    # Featurize
    X, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, \
    visible_list_list, masked_list_list, masked_chain_length_list_list, \
    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
    pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch_clones, device, None, None, None, None, None, None, ca_only=False
    )

    # Get native sequence
    native_seq_idx = S[0].cpu().numpy()
    seq_len = int(mask[0].sum().item())
    native_seq_idx = native_seq_idx[:seq_len]
    native_seq = indices_to_sequences([native_seq_idx])[0]

    print(f"\nProtein: {Path(pdb_path).stem}")
    print(f"Length: {seq_len} residues")
    print(f"Native sequence: {native_seq}")

    # Generate sequences
    print(f"\nGenerating {num_samples} sequences...")

    designed_sequences = []
    designed_indices = []
    log_probs_list = []
    recoveries = []

    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Random seed for variety
            randn = torch.randn(chain_M.shape, device=device) * temperature

            # Forward pass
            log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)

            # Get designed sequence (argmax)
            designed_idx = torch.argmax(log_probs, dim=-1)[0].cpu().numpy()
            designed_idx = designed_idx[:seq_len]

            # Convert to sequence
            designed_seq = indices_to_sequences([designed_idx])[0]

            # Calculate recovery
            recovery = np.mean(designed_idx == native_seq_idx) * 100

            designed_sequences.append(designed_seq)
            designed_indices.append(designed_idx)
            recoveries.append(recovery)
            log_probs_list.append(log_probs[0, :seq_len].cpu().numpy())

            print(f"  Sample {sample_idx + 1}: {recovery:.1f}% recovery")

    # Calculate statistics
    mean_recovery = np.mean(recoveries)
    std_recovery = np.std(recoveries)

    # Position-wise recovery
    position_recovery = np.zeros(seq_len)
    for designed_idx in designed_indices:
        position_recovery += (designed_idx == native_seq_idx)
    position_recovery = position_recovery / num_samples * 100

    # Consensus sequence
    consensus_idx = np.array([
        np.bincount([designed_indices[i][j] for i in range(num_samples)]).argmax()
        for j in range(seq_len)
    ])
    consensus_seq = indices_to_sequences([consensus_idx])[0]
    consensus_recovery = np.mean(consensus_idx == native_seq_idx) * 100

    print(f"\n  Mean recovery: {mean_recovery:.1f} ± {std_recovery:.1f}%")
    print(f"  Consensus recovery: {consensus_recovery:.1f}%")

    results = {
        'config': config,
        'native_sequence': native_seq,
        'designed_sequences': designed_sequences,
        'consensus_sequence': consensus_seq,
        'recoveries': recoveries,
        'mean_recovery': float(mean_recovery),
        'std_recovery': float(std_recovery),
        'consensus_recovery': float(consensus_recovery),
        'position_recovery': position_recovery.tolist(),
        'sequence_length': seq_len
    }

    return results

# Run all variants
print("\nRunning all variants...")
all_results = {}

for variant in variants:
    try:
        results = run_variant(variant, pdb_path, num_samples=10)
        all_results[variant['name']] = results

        # Save individual variant output
        variant_dir = output_dir / variant['name']
        variant_dir.mkdir(exist_ok=True)

        # Save FASTA file
        fasta_file = variant_dir / 'sequences.fasta'
        with open(fasta_file, 'w') as f:
            # Native sequence
            f.write(f">Native_sequence\n{results['native_sequence']}\n\n")

            # Designed sequences
            for i, (seq, recovery) in enumerate(zip(results['designed_sequences'],
                                                     results['recoveries'])):
                f.write(f">Designed_sample_{i+1}_recovery_{recovery:.1f}%\n{seq}\n\n")

            # Consensus
            f.write(f">Consensus_recovery_{results['consensus_recovery']:.1f}%\n")
            f.write(f"{results['consensus_sequence']}\n")

        # Save detailed JSON
        json_file = variant_dir / 'results.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save comparison text
        comparison_file = variant_dir / 'comparison.txt'
        with open(comparison_file, 'w') as f:
            f.write(f"Variant: {variant['name']}\n")
            f.write(f"Description: {variant['description']}\n")
            f.write(f"="*80 + "\n\n")

            f.write("SEQUENCE COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"Native:    {results['native_sequence']}\n\n")

            for i, (seq, recovery) in enumerate(zip(results['designed_sequences'][:3],
                                                     results['recoveries'][:3])):
                f.write(f"Sample {i+1}:  {seq}  ({recovery:.1f}% recovery)\n")

            f.write(f"\nConsensus: {results['consensus_sequence']}  ")
            f.write(f"({results['consensus_recovery']:.1f}% recovery)\n\n")

            f.write("POSITION-WISE RECOVERY\n")
            f.write("-"*80 + "\n")
            pos_rec = np.array(results['position_recovery'])
            f.write(f"Mean: {pos_rec.mean():.1f}%\n")
            f.write(f"Min:  {pos_rec.min():.1f}%\n")
            f.write(f"Max:  {pos_rec.max():.1f}%\n")

            # Show positions with low recovery
            low_recovery_positions = np.where(pos_rec < 50)[0]
            if len(low_recovery_positions) > 0:
                f.write(f"\nPositions with <50% recovery: {len(low_recovery_positions)}\n")
                for pos in low_recovery_positions[:10]:  # Show first 10
                    f.write(f"  Position {pos+1}: {pos_rec[pos]:.0f}% ")
                    f.write(f"(native: {results['native_sequence'][pos]})\n")

        print(f"  ✅ Saved to: {variant_dir}/")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_results[variant['name']] = {'error': str(e)}

# Create summary comparison
print(f"\n{'='*80}")
print("CREATING SUMMARY COMPARISON")
print("="*80)

summary_file = output_dir / 'SUMMARY.txt'
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL VARIANTS SEQUENCE OUTPUT SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write(f"Test protein: {Path(pdb_path).stem}\n")
    f.write(f"Number of samples per variant: 10\n")
    f.write(f"Date: 2026-02-04\n\n")

    # Get native sequence
    if 'Baseline' in all_results and 'error' not in all_results['Baseline']:
        native = all_results['Baseline']['native_sequence']
        f.write(f"Native sequence ({len(native)} residues):\n")
        f.write(f"{native}\n\n")

    f.write("="*80 + "\n")
    f.write("RECOVERY COMPARISON\n")
    f.write("="*80 + "\n\n")

    f.write(f"{'Variant':<15} {'Mean Recovery':<15} {'Consensus Recovery':<20} {'Status'}\n")
    f.write("-"*80 + "\n")

    for variant in variants:
        name = variant['name']
        if name in all_results and 'error' not in all_results[name]:
            results = all_results[name]
            mean_rec = results['mean_recovery']
            cons_rec = results['consensus_recovery']
            status = "✅"
            f.write(f"{name:<15} {mean_rec:>6.1f}%        {cons_rec:>6.1f}%              {status}\n")
        else:
            f.write(f"{name:<15} {'ERROR':<15} {'ERROR':<20} ❌\n")

    f.write("\n" + "="*80 + "\n")
    f.write("CONSENSUS SEQUENCES\n")
    f.write("="*80 + "\n\n")

    for variant in variants:
        name = variant['name']
        if name in all_results and 'error' not in all_results[name]:
            results = all_results[name]
            f.write(f"{name}:\n")
            f.write(f"{results['consensus_sequence']}\n")
            f.write(f"Recovery: {results['consensus_recovery']:.1f}%\n\n")

    f.write("="*80 + "\n")
    f.write("DETAILED DIFFERENCES (Consensus vs Native)\n")
    f.write("="*80 + "\n\n")

    if 'Baseline' in all_results and 'error' not in all_results['Baseline']:
        native = all_results['Baseline']['native_sequence']

        for variant in variants:
            name = variant['name']
            if name in all_results and 'error' not in all_results[name]:
                results = all_results[name]
                consensus = results['consensus_sequence']

                # Find differences
                differences = []
                for i, (n, c) in enumerate(zip(native, consensus)):
                    if n != c:
                        differences.append((i+1, n, c))

                f.write(f"{name}: {len(differences)} differences\n")
                if differences:
                    for pos, native_aa, designed_aa in differences[:10]:  # Show first 10
                        f.write(f"  Position {pos}: {native_aa} → {designed_aa}\n")
                    if len(differences) > 10:
                        f.write(f"  ... and {len(differences) - 10} more\n")
                f.write("\n")

print(f"\n✅ Summary saved to: {summary_file}")

# Save all results JSON
all_results_file = output_dir / 'all_results.json'
with open(all_results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"✅ Complete results saved to: {all_results_file}")

print(f"\n{'='*80}")
print("ALL VARIANTS COMPLETE")
print("="*80)

print(f"\nOutput directory: {output_dir}/")
print("\nGenerated files:")
print(f"  - SUMMARY.txt - Overall comparison")
print(f"  - all_results.json - Complete results")
print(f"  - [Variant]/ - Individual variant outputs:")
print(f"      - sequences.fasta - All sequences in FASTA format")
print(f"      - results.json - Detailed results")
print(f"      - comparison.txt - Human-readable comparison")

print("\nVariants tested:")
for variant in variants:
    name = variant['name']
    if name in all_results and 'error' not in all_results[name]:
        recovery = all_results[name]['mean_recovery']
        print(f"  ✅ {name:<15} {recovery:>6.1f}% mean recovery")
    else:
        print(f"  ❌ {name:<15} ERROR")
