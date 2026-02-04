"""
Reference Implementation with Real Protein Feature Extraction

This file demonstrates what a COMPLETE, production-ready implementation
would look like, contrasted with the simplified demonstrations in other files.

KEY DIFFERENCES FROM OTHER IMPLEMENTATIONS:
1. Real protein feature extraction (RBF, orientations, etc.)
2. Proper graph construction from coordinates
3. Complete encoder-decoder architecture
4. Actual timing and profiling infrastructure

This serves as a reference for what would be needed to validate benchmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
import time


# ============================================================================
# REAL Protein Feature Extraction (Not Placeholder)
# ============================================================================

def rbf_encode_distances(distances: torch.Tensor, d_min: float = 0.0,
                         d_max: float = 20.0, d_count: int = 16) -> torch.Tensor:
    """
    REAL Radial Basis Function encoding of distances.

    This is what should be used instead of torch.randn() for distance features.

    Args:
        distances: [N, N] or [E] pairwise distances in Angstroms
        d_min: Minimum distance for RBF centers
        d_max: Maximum distance for RBF centers
        d_count: Number of RBF centers

    Returns:
        RBF encoded distances [N, N, d_count] or [E, d_count]
    """
    # Create RBF centers
    d_mu = torch.linspace(d_min, d_max, d_count, device=distances.device)
    d_sigma = (d_max - d_min) / d_count

    # Compute RBF features
    # distances[..., None] broadcasts to [..., 1]
    # d_mu broadcasts to [1, 1, ..., d_count]
    rbf = torch.exp(-((distances.unsqueeze(-1) - d_mu) ** 2) / (2 * d_sigma ** 2))

    return rbf


def build_knn_graph(coords: torch.Tensor, k: int = 30,
                    self_loops: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    REAL k-NN graph construction from CA coordinates.

    This is what should be used instead of assuming edge_index exists.

    Args:
        coords: [N, 3] CA coordinates
        k: Number of nearest neighbors
        self_loops: Whether to include self-connections

    Returns:
        edge_index: [2, E] edge connectivity
        distances: [E] edge distances
    """
    N = coords.shape[0]

    # Compute pairwise distance matrix
    # This is the O(N²) operation mentioned in the reference document
    dist_matrix = torch.cdist(coords, coords)  # [N, N]

    if not self_loops:
        # Mask self-connections with large value
        dist_matrix = dist_matrix + torch.eye(N, device=coords.device) * 1e6

    # Get k nearest neighbors for each node
    # topk returns (values, indices)
    k_actual = min(k, N - 1) if not self_loops else min(k, N)
    nearest_dists, nearest_indices = torch.topk(
        dist_matrix, k_actual, dim=1, largest=False
    )  # [N, k]

    # Build edge index
    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(-1, k_actual)  # [N, k]
    dst = nearest_indices  # [N, k]

    edge_index = torch.stack([src.flatten(), dst.flatten()])  # [2, N*k]
    distances = nearest_dists.flatten()  # [N*k]

    return edge_index, distances


def compute_orientations(coords: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
    """
    REAL orientation features from coordinates.

    Computes relative orientations between connected residues.
    This is typically part of the edge features in ProteinMPNN.

    Args:
        coords: [N, 3] CA coordinates
        edge_index: [2, E] edge connectivity

    Returns:
        orientations: [E, 3] unit vectors along edges
    """
    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    # Get source and destination coordinates
    src_coords = coords[src_idx]  # [E, 3]
    dst_coords = coords[dst_idx]  # [E, 3]

    # Compute displacement vectors
    displacements = dst_coords - src_coords  # [E, 3]

    # Normalize to unit vectors
    norms = torch.norm(displacements, dim=-1, keepdim=True)  # [E, 1]
    orientations = displacements / (norms + 1e-8)  # [E, 3]

    return orientations


def extract_protein_features(coords: torch.Tensor,
                             k: int = 30) -> Dict[str, torch.Tensor]:
    """
    COMPLETE protein feature extraction pipeline.

    This is what's ACTUALLY needed (not torch.randn placeholders).

    Args:
        coords: [N, 3] CA coordinates

    Returns:
        Dictionary with:
        - node_features: [N, feature_dim] per-residue features
        - edge_index: [2, E] connectivity
        - edge_features: [E, edge_dim] edge features
    """
    N = coords.shape[0]

    # Build graph
    edge_index, distances = build_knn_graph(coords, k=k)

    # Node features (positional encoding)
    # Simple positional encoding (real ProteinMPNN would include amino acid type)
    positions = torch.arange(N, device=coords.device, dtype=torch.float32)
    pos_encoding = torch.stack([
        torch.sin(positions / 10000 ** (2 * i / 16))
        for i in range(8)
    ] + [
        torch.cos(positions / 10000 ** (2 * i / 16))
        for i in range(8)
    ], dim=-1)  # [N, 16]

    # Distances to center of mass (simple geometric feature)
    center_of_mass = coords.mean(dim=0, keepdim=True)  # [1, 3]
    dist_to_com = torch.norm(coords - center_of_mass, dim=-1, keepdim=True)  # [N, 1]

    # Combine node features
    node_features = torch.cat([pos_encoding, dist_to_com], dim=-1)  # [N, 17]

    # Edge features
    rbf_distances = rbf_encode_distances(distances)  # [E, 16]
    orientations = compute_orientations(coords, edge_index)  # [E, 3]

    edge_features = torch.cat([rbf_distances, orientations], dim=-1)  # [E, 19]

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'distances': distances
    }


# ============================================================================
# REAL Model with Complete Architecture
# ============================================================================

class RealMPNNLayer(nn.Module):
    """
    COMPLETE message-passing layer (not simplified demo).

    This includes proper message computation, aggregation, and update.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Message network: f_msg(h_i, h_j, e_ij)
        self.W_msg = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update network: f_update(h_i, m_i)
        self.W_update = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        Real message passing implementation.

        Args:
            node_features: [N, node_dim]
            edge_index: [2, E]
            edge_features: [E, edge_dim]

        Returns:
            Updated node_features: [N, node_dim]
        """
        N = node_features.shape[0]
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # Gather source and destination node features
        src_features = node_features[src_idx]  # [E, node_dim]
        dst_features = node_features[dst_idx]  # [E, node_dim]

        # Compute messages
        message_input = torch.cat([src_features, dst_features, edge_features], dim=-1)
        messages = self.W_msg(message_input)  # [E, hidden_dim]

        # Aggregate messages by destination node (scatter_add)
        aggregated = torch.zeros(N, self.hidden_dim, device=node_features.device)
        aggregated.index_add_(0, dst_idx, messages)  # In-place scatter_add

        # Update node features
        update_input = torch.cat([node_features, aggregated], dim=-1)
        updates = self.W_update(update_input)  # [N, node_dim]

        # Residual connection + layer norm
        return self.layer_norm(node_features + updates)


class RealProteinMPNN(nn.Module):
    """
    COMPLETE ProteinMPNN with real feature extraction.

    This is what would be needed for actual benchmarking.
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 vocab_size: int = 20,
                 k_neighbors: int = 30):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.vocab_size = vocab_size

        # Input projections (from real features to hidden_dim)
        # Real features have specific dimensions from extract_protein_features
        self.node_proj = nn.Linear(17, hidden_dim)  # 17 from positional + geometric features
        self.edge_proj = nn.Linear(19, hidden_dim)  # 19 from RBF + orientations

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            RealMPNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_encoder_layers)
        ])

        # Decoder (simplified - real version would be autoregressive transformer)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        COMPLETE forward pass from coordinates to sequence.

        Args:
            coords: [N, 3] CA coordinates in Angstroms

        Returns:
            logits: [N, vocab_size] amino acid predictions
        """
        # Step 1: REAL feature extraction (not placeholders)
        features = extract_protein_features(coords, k=self.k_neighbors)

        # Step 2: Project to hidden dimension
        node_h = self.node_proj(features['node_features'])  # [N, hidden_dim]
        edge_h = self.edge_proj(features['edge_features'])  # [E, hidden_dim]

        # Step 3: Encoder
        for layer in self.encoder_layers:
            node_h = layer(node_h, features['edge_index'], edge_h)

        # Step 4: Decoder
        # Add batch dimension for TransformerEncoderLayer
        node_h = node_h.unsqueeze(0)  # [1, N, hidden_dim]

        for layer in self.decoder_layers:
            node_h = layer(node_h)

        node_h = node_h.squeeze(0)  # [N, hidden_dim]

        # Step 5: Output projection
        logits = self.output_proj(node_h)  # [N, vocab_size]

        return logits


# ============================================================================
# REAL Benchmarking Infrastructure
# ============================================================================

class RealBenchmark:
    """
    ACTUAL benchmarking with timing and profiling.

    This is what would be needed to validate reported speedups.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def benchmark_inference(self,
                          coords: torch.Tensor,
                          num_runs: int = 100,
                          warmup_runs: int = 10) -> Dict[str, float]:
        """
        REAL timing measurements with proper synchronization.

        Args:
            coords: [N, 3] protein coordinates
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs (excluded from timing)

        Returns:
            Dictionary with mean_time, std_time, throughput
        """
        coords = coords.to(self.device)

        # Warmup (critical for GPU benchmarking)
        print(f"Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(coords)
            if self.device.type in ['cuda', 'mps']:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()

        # Actual timing
        print(f"Benchmarking ({num_runs} runs)...")
        times = []

        for i in range(num_runs):
            # Start timer
            if self.device.type in ['cuda', 'mps']:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()

            start = time.perf_counter()

            # Inference
            with torch.no_grad():
                _ = self.model(coords)

            # End timer (ensure GPU completion)
            if self.device.type in ['cuda', 'mps']:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()

            end = time.perf_counter()

            times.append(end - start)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{num_runs} runs")

        # Compute statistics
        import numpy as np
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        seq_length = coords.shape[0]
        throughput = seq_length / mean_time

        return {
            'mean_time_sec': mean_time,
            'std_time_sec': std_time,
            'mean_time_ms': mean_time * 1000,
            'std_time_ms': std_time * 1000,
            'throughput_res_per_sec': throughput,
            'sequence_length': seq_length,
            'num_runs': num_runs
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Reference Implementation - Real Feature Extraction & Benchmarking")
    print("="*70)

    # Create realistic protein coordinates (100-residue helix)
    N = 100
    # Simple helix coordinates (3.6 residues per turn, 1.5 Å rise per residue)
    t = torch.linspace(0, N * 2 * math.pi / 3.6, N)
    coords = torch.stack([
        5.0 * torch.cos(t),
        5.0 * torch.sin(t),
        1.5 * torch.arange(N, dtype=torch.float32)
    ], dim=1)  # [N, 3]

    print(f"\nTest protein: {N} residues (α-helix)")
    print(f"Coordinate range: {coords.min().item():.2f} to {coords.max().item():.2f} Å")

    # Extract features (REAL, not placeholder)
    print("\n" + "="*70)
    print("Feature Extraction (REAL implementation)")
    print("="*70)
    features = extract_protein_features(coords, k=30)

    print(f"Node features shape: {features['node_features'].shape}")
    print(f"Edge index shape: {features['edge_index'].shape}")
    print(f"Edge features shape: {features['edge_features'].shape}")
    print(f"Number of edges: {features['edge_index'].shape[1]}")
    print(f"Average degree: {features['edge_index'].shape[1] / N:.1f}")

    # Create model
    print("\n" + "="*70)
    print("Model Creation (COMPLETE architecture)")
    print("="*70)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = RealProteinMPNN(hidden_dim=128, num_encoder_layers=3)
    model = model.to(device)
    coords = coords.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Forward pass
    print("\n" + "="*70)
    print("Forward Pass (End-to-end)")
    print("="*70)
    with torch.no_grad():
        logits = model(coords)

    print(f"Output shape: {logits.shape}")
    print(f"Output range: {logits.min().item():.3f} to {logits.max().item():.3f}")

    # Sample sequence
    sequence = torch.argmax(logits, dim=-1)
    print(f"Predicted sequence (first 20): {sequence[:20].tolist()}")

    # Benchmark (small scale for demonstration)
    print("\n" + "="*70)
    print("Benchmarking (REAL timing)")
    print("="*70)
    benchmark = RealBenchmark(model, device)
    results = benchmark.benchmark_inference(coords, num_runs=20, warmup_runs=5)

    print(f"\nResults:")
    print(f"  Mean time: {results['mean_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_res_per_sec']:.1f} residues/sec")
    print(f"  Sequence length: {results['sequence_length']}")

    print("\n" + "="*70)
    print("IMPORTANT: These are REAL measurements, not simulations!")
    print("However, this is still a simplified model compared to full ProteinMPNN.")
    print("="*70)
