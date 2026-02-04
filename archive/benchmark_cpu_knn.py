#!/usr/bin/env python3
"""
Test Zero-Copy Graph Construction Optimization
Use CPU for k-NN search, leverage unified memory to avoid GPU-CPU copies

From expert_proteinmpnn.txt optimization #3:
- Perform k-NN search on CPU (better for divergent branch logic)
- Use unified memory (no copy overhead on Apple Silicon)
- GPU reads indices directly from shared memory
"""

import torch
import numpy as np
import time
import json
import sys
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '/Users/ktretina/claude_dir/ProteinMPNN')

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

device = torch.device("mps")

def cpu_knn_search(X_ca, k_neighbors, mask):
    """
    Perform k-NN search on CPU using sklearn.

    X_ca: [B, L, 3] Ca coordinates
    Returns: [B, L, k] neighbor indices
    """
    B, L, _ = X_ca.shape

    # Move to CPU and numpy
    X_np = X_ca.cpu().numpy()
    mask_np = mask.cpu().numpy()

    E_idx_list = []

    for b in range(B):
        # Get valid positions
        valid_mask = mask_np[b] > 0
        valid_coords = X_np[b][valid_mask]
        n_valid = valid_coords.shape[0]

        if n_valid == 0:
            # Empty sequence, return zeros
            E_idx_list.append(torch.zeros(L, k_neighbors, dtype=torch.long))
            continue

        # Use sklearn NearestNeighbors (optimized for CPU)
        k_actual = min(k_neighbors, n_valid)
        nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm='ball_tree').fit(valid_coords)

        # Find neighbors for all valid positions
        distances, indices = nbrs.kneighbors(valid_coords)

        # Map back to original indexing
        valid_indices = np.where(valid_mask)[0]
        E_idx_batch = np.zeros((L, k_neighbors), dtype=np.int64)

        for i, orig_idx in enumerate(valid_indices):
            neighbor_orig_indices = valid_indices[indices[i]]
            # Pad if needed
            if len(neighbor_orig_indices) < k_neighbors:
                neighbor_orig_indices = np.pad(
                    neighbor_orig_indices,
                    (0, k_neighbors - len(neighbor_orig_indices)),
                    mode='edge'
                )
            E_idx_batch[orig_idx] = neighbor_orig_indices

        E_idx_list.append(torch.from_numpy(E_idx_batch))

    E_idx = torch.stack(E_idx_list, dim=0)

    # Move to GPU (on unified memory, this should be fast)
    return E_idx.to(device)

class ProteinMPNN_CPUkNN(ProteinMPNN):
    """Modified ProteinMPNN that uses CPU for k-NN search."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_cpu_knn = True

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=False, decoding_order=None):
        """Modified forward that uses CPU k-NN."""
        device = X.device

        # Original feature extraction (without k-NN)
        # We need to manually compute features with CPU k-NN

        # For simplicity, just call parent forward
        # This is a proof of concept - full implementation would require
        # modifying the ProteinFeatures class

        return super().forward(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order, decoding_order)

def load_model_standard(model_path):
    """Load baseline model with GPU k-NN."""
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

def benchmark_cpu_knn_only(pdb_path, k_neighbors=48, num_runs=20):
    """
    Benchmark just the k-NN search portion on CPU vs GPU.
    This isolates the graph construction cost.
    """

    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]

    X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
        [protein], device, None, None, None, None, None, None, ca_only=False
    )

    # Extract Ca coordinates
    X_ca = X[:, :, 1, :]  # [B, L, 3]
    B, L, _ = X_ca.shape

    print(f"\nProtein size: {L} residues")
    print(f"k-neighbors: {k_neighbors}")

    # Benchmark CPU k-NN
    print("\nBenchmarking CPU k-NN search...")
    cpu_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        E_idx_cpu = cpu_knn_search(X_ca, k_neighbors, mask)
        torch.mps.synchronize()
        end = time.perf_counter()
        cpu_times.append(end - start)

    cpu_mean = np.mean(cpu_times) * 1000
    cpu_std = np.std(cpu_times) * 1000

    print(f"CPU k-NN: {cpu_mean:.2f} ± {cpu_std:.2f} ms")

    # Benchmark GPU k-NN (PyTorch pairwise distance + topk)
    print("\nBenchmarking GPU k-NN search...")
    gpu_times = []
    for _ in range(num_runs):
        torch.mps.synchronize()
        start = time.perf_counter()

        # Compute pairwise distances (GPU)
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X_ca, 1) - torch.unsqueeze(X_ca, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + 1e-6)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max

        # TopK (GPU)
        D_neighbors, E_idx_gpu = torch.topk(D_adjust, min(k_neighbors, L), dim=-1, largest=False)

        torch.mps.synchronize()
        end = time.perf_counter()
        gpu_times.append(end - start)

    gpu_mean = np.mean(gpu_times) * 1000
    gpu_std = np.std(gpu_times) * 1000

    print(f"GPU k-NN: {gpu_mean:.2f} ± {gpu_std:.2f} ms")

    speedup = gpu_mean / cpu_mean

    return {
        'cpu_knn_ms': cpu_mean,
        'cpu_std_ms': cpu_std,
        'gpu_knn_ms': gpu_mean,
        'gpu_std_ms': gpu_std,
        'speedup': speedup,
        'protein_length': L,
        'k_neighbors': k_neighbors
    }

pdb_path = '/Users/ktretina/claude_dir/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb'
model_path = '/Users/ktretina/claude_dir/ProteinMPNN/vanilla_model_weights/v_48_020.pt'

print("="*70)
print("ZERO-COPY GRAPH CONSTRUCTION TEST")
print("Expert optimization #3: CPU k-NN with unified memory")
print("="*70)

# Test just the k-NN portion
print("\n" + "-"*70)
print("TESTING K-NN SEARCH ONLY (Isolated)")
print("-"*70)

result = benchmark_cpu_knn_only(pdb_path, k_neighbors=48, num_runs=20)

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print(f"\nProtein: {result['protein_length']} residues")
print(f"k-neighbors: {result['k_neighbors']}")
print(f"\nCPU k-NN: {result['cpu_knn_ms']:.2f} ± {result['cpu_std_ms']:.2f} ms")
print(f"GPU k-NN: {result['gpu_knn_ms']:.2f} ± {result['gpu_std_ms']:.2f} ms")
print(f"\nSpeedup: {result['speedup']:.2f}x")

if result['speedup'] > 1.1:
    print("\n✅ CPU k-NN is faster!")
    print("   Unified memory architecture benefits CPU search")
    print("   Recommendation: Use CPU k-NN for graph construction")
elif result['speedup'] > 0.9:
    print("\n⚠️  CPU and GPU k-NN have similar performance")
    print("   No significant benefit from unified memory approach")
else:
    print("\n❌ GPU k-NN is faster")
    print("   GPU parallelism outweighs branch prediction benefits")
    print("   Recommendation: Keep GPU k-NN")

# Note about full model integration
print("\n" + "-"*70)
print("NOTE: Full Model Integration")
print("-"*70)
print("Full integration would require modifying ProteinFeatures class")
print("to accept pre-computed E_idx from CPU k-NN search.")
print("This benchmark isolates just the k-NN component.")

# Save results
Path('output').mkdir(exist_ok=True)
with open('output/cpu_knn_benchmark.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*70)
print("Results saved to: output/cpu_knn_benchmark.json")
print("="*70)
