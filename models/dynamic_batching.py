"""
Dynamic Batching with Length Sorting for ProteinMPNN

Optimization: Intelligent batching to minimize padding waste.

Key benefits:
- 2-4x throughput improvement
- Minimizes wasted computation on padding
- Adaptive batch sizing based on protein length
- Better memory utilization

Reference: Section 8 of optimization document
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
import math


@dataclass
class ProteinBatch:
    """Container for batched protein data."""
    coords: torch.Tensor  # [batch_size, max_len, 3 or 6]
    edge_indices: List[torch.Tensor]  # List of edge indices per protein
    edge_features: List[torch.Tensor]  # List of edge features per protein
    lengths: torch.Tensor  # [batch_size] actual lengths
    padding_mask: torch.Tensor  # [batch_size, max_len] True = padding
    native_sequences: Optional[List[str]] = None


class LengthBasedBatcher:
    """
    Intelligent batcher that groups proteins by length to minimize padding.

    Creates buckets of similar-length proteins and computes optimal batch sizes
    to maximize memory utilization without overflow.
    """

    def __init__(
        self,
        bucket_boundaries: Optional[List[int]] = None,
        max_tokens_per_batch: int = 8192,
        encoder_memory_factor: float = 1.0,
        decoder_memory_factor: float = 2.0
    ):
        """
        Args:
            bucket_boundaries: Length boundaries for buckets (e.g., [50, 100, 200, 500])
            max_tokens_per_batch: Maximum tokens (residues) per batch
            encoder_memory_factor: Memory scaling for encoder (O(L))
            decoder_memory_factor: Memory scaling for decoder (O(L) with cache)
        """
        if bucket_boundaries is None:
            # Default buckets
            bucket_boundaries = [50, 100, 150, 200, 300, 500, 1000]

        self.bucket_boundaries = sorted(bucket_boundaries)
        self.max_tokens = max_tokens_per_batch
        self.encoder_factor = encoder_memory_factor
        self.decoder_factor = decoder_memory_factor

    def create_buckets(
        self,
        protein_data: List[Dict]
    ) -> Dict[int, List[Dict]]:
        """
        Sort proteins into length-based buckets.

        Args:
            protein_data: List of dicts with 'coords', 'length', etc.

        Returns:
            Dictionary mapping bucket_id to list of proteins
        """
        buckets = {i: [] for i in range(len(self.bucket_boundaries) + 1)}

        for protein in protein_data:
            length = protein['length']

            # Find appropriate bucket
            bucket_id = len(self.bucket_boundaries)  # Default to last bucket
            for i, boundary in enumerate(self.bucket_boundaries):
                if length <= boundary:
                    bucket_id = i
                    break

            buckets[bucket_id].append(protein)

        # Remove empty buckets
        buckets = {k: v for k, v in buckets.items() if v}

        # Print bucket statistics
        print("Bucket Statistics:")
        for bucket_id, proteins in buckets.items():
            if bucket_id < len(self.bucket_boundaries):
                max_len = self.bucket_boundaries[bucket_id]
                min_len = self.bucket_boundaries[bucket_id - 1] if bucket_id > 0 else 0
                print(f"  Bucket {bucket_id} ({min_len}-{max_len}): {len(proteins)} proteins")
            else:
                print(f"  Bucket {bucket_id} (>{self.bucket_boundaries[-1]}): {len(proteins)} proteins")

        return buckets

    def compute_batch_size(
        self,
        length: int,
        mode: str = 'balanced'
    ) -> int:
        """
        Compute optimal batch size for given protein length.

        Args:
            length: Protein length
            mode: 'encoder', 'decoder', or 'balanced'

        Returns:
            Optimal batch size
        """
        if mode == 'encoder':
            # Encoder: O(L * E) where E = edges ≈ k*L
            # Memory ≈ B * L * L (for adjacency)
            # So B ≈ max_tokens / L²
            batch_size = max(1, int(self.max_tokens / (length * length * self.encoder_factor)))
        elif mode == 'decoder':
            # Decoder: O(L) with KV cache
            # Memory ≈ B * L
            batch_size = max(1, int(self.max_tokens / (length * self.decoder_factor)))
        else:  # balanced
            # Use decoder constraint (typically tighter)
            batch_size = max(1, int(self.max_tokens / (length * self.decoder_factor)))

        return min(batch_size, 32)  # Cap at 32 for practical reasons

    def create_batches(
        self,
        bucket: List[Dict],
        batch_size: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Create batches from a bucket.

        Args:
            bucket: List of proteins in same length bucket
            batch_size: Fixed batch size (if None, compute adaptively)

        Returns:
            List of batches (each batch is a list of proteins)
        """
        if not bucket:
            return []

        # Compute batch size if not provided
        if batch_size is None:
            avg_length = np.mean([p['length'] for p in bucket])
            batch_size = self.compute_batch_size(int(avg_length))

        # Split into batches
        batches = []
        for i in range(0, len(bucket), batch_size):
            batch = bucket[i:i + batch_size]
            batches.append(batch)

        return batches


class PaddingEfficiencyTracker:
    """Track padding efficiency metrics."""

    def __init__(self):
        self.total_tokens = 0
        self.padding_tokens = 0
        self.batches_processed = 0

    def update(self, batch: ProteinBatch):
        """Update statistics from a batch."""
        batch_size, max_len = batch.coords.shape[:2]
        total = batch_size * max_len
        actual = batch.lengths.sum().item()

        self.total_tokens += total
        self.padding_tokens += (total - actual)
        self.batches_processed += 1

    def get_efficiency(self) -> float:
        """Get padding efficiency (1.0 = no padding, 0.0 = all padding)."""
        if self.total_tokens == 0:
            return 1.0
        return 1.0 - (self.padding_tokens / self.total_tokens)

    def __repr__(self):
        eff = self.get_efficiency()
        return (f"PaddingEfficiency(batches={self.batches_processed}, "
                f"efficiency={eff:.1%}, "
                f"wasted={self.padding_tokens}/{self.total_tokens})")


class DynamicBatchCollator:
    """
    Collate function for creating batched tensors with minimal padding.

    Pads proteins to the maximum length in the batch (not global maximum).
    """

    def __init__(
        self,
        pad_value: float = 0.0,
        return_native_sequences: bool = False
    ):
        self.pad_value = pad_value
        self.return_native_sequences = return_native_sequences

    def __call__(self, batch: List[Dict]) -> ProteinBatch:
        """
        Collate a batch of proteins.

        Args:
            batch: List of protein dictionaries

        Returns:
            ProteinBatch with padded tensors
        """
        batch_size = len(batch)

        # Find max length in this batch
        lengths = torch.tensor([p['length'] for p in batch])
        max_len = lengths.max().item()

        # Get coordinate dimension
        coord_dim = batch[0]['coords'].shape[-1]

        # Initialize padded tensor
        coords = torch.full(
            (batch_size, max_len, coord_dim),
            self.pad_value,
            dtype=batch[0]['coords'].dtype
        )

        # Create padding mask
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

        # Fill in actual data
        for i, protein in enumerate(batch):
            length = protein['length']
            coords[i, :length] = protein['coords']
            padding_mask[i, :length] = False

        # Collect edges and features (not padded, kept as list)
        edge_indices = [p['edge_index'] for p in batch]
        edge_features = [p['edge_features'] for p in batch]

        # Collect native sequences if available
        native_sequences = None
        if self.return_native_sequences and 'native_sequence' in batch[0]:
            native_sequences = [p['native_sequence'] for p in batch]

        return ProteinBatch(
            coords=coords,
            edge_indices=edge_indices,
            edge_features=edge_features,
            lengths=lengths,
            padding_mask=padding_mask,
            native_sequences=native_sequences
        )


class DynamicBatchedProteinMPNN(nn.Module):
    """
    ProteinMPNN with dynamic batching support.

    Processes variable-length proteins efficiently with minimal padding waste.
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_tokens_per_batch: int = 8192,
        bucket_boundaries: Optional[List[int]] = None
    ):
        """
        Args:
            base_model: Base ProteinMPNN model
            max_tokens_per_batch: Maximum tokens per batch
            bucket_boundaries: Length boundaries for buckets
        """
        super().__init__()
        self.model = base_model
        self.batcher = LengthBasedBatcher(
            bucket_boundaries=bucket_boundaries,
            max_tokens_per_batch=max_tokens_per_batch
        )
        self.collator = DynamicBatchCollator()
        self.efficiency_tracker = PaddingEfficiencyTracker()

    def forward_batch(
        self,
        batch: ProteinBatch,
        temperature: float = 0.1
    ) -> List[torch.Tensor]:
        """
        Forward pass on a dynamic batch.

        Args:
            batch: ProteinBatch with padded data
            temperature: Sampling temperature

        Returns:
            List of sequences (one per protein, variable length)
        """
        # Track efficiency
        self.efficiency_tracker.update(batch)

        # Process each protein individually (since edges are different)
        # In practice, could use batched GNN operations
        sequences = []

        for i in range(len(batch.lengths)):
            length = batch.lengths[i].item()
            coords = batch.coords[i:i+1, :length]  # [1, length, dim]
            edge_index = batch.edge_indices[i]
            edge_features = batch.edge_features[i]

            # Forward through model
            seq = self.model(coords, edge_index, edge_features, temperature=temperature)
            sequences.append(seq.squeeze(0))  # Remove batch dim

        return sequences

    def process_dataset(
        self,
        proteins: List[Dict],
        batch_size: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> List[torch.Tensor]:
        """
        Process entire dataset with optimal batching.

        Args:
            proteins: List of protein data
            batch_size: Fixed batch size (None = adaptive)
            callback: Optional callback(batch_idx, sequences)

        Returns:
            List of all generated sequences
        """
        # Create buckets
        buckets = self.batcher.create_buckets(proteins)

        all_sequences = []
        batch_idx = 0

        # Process each bucket
        for bucket_id, bucket in buckets.items():
            print(f"\nProcessing bucket {bucket_id} ({len(bucket)} proteins)...")

            # Create batches for this bucket
            batches = self.batcher.create_batches(bucket, batch_size)

            print(f"  Created {len(batches)} batches")

            # Process each batch
            for batch_proteins in batches:
                # Collate batch
                batch = self.collator(batch_proteins)

                # Forward pass
                sequences = self.forward_batch(batch)
                all_sequences.extend(sequences)

                # Callback
                if callback:
                    callback(batch_idx, sequences)

                batch_idx += 1

        # Print efficiency statistics
        print(f"\n{self.efficiency_tracker}")

        return all_sequences

    def get_efficiency_stats(self) -> Dict:
        """Get padding efficiency statistics."""
        return {
            'efficiency': self.efficiency_tracker.get_efficiency(),
            'total_tokens': self.efficiency_tracker.total_tokens,
            'padding_tokens': self.efficiency_tracker.padding_tokens,
            'batches': self.efficiency_tracker.batches_processed
        }


def compare_batching_strategies():
    """Compare naive vs dynamic batching."""
    print("Comparing Batching Strategies\n")
    print("="*60)

    # Generate synthetic proteins with diverse lengths
    np.random.seed(42)
    lengths = np.random.choice([50, 75, 100, 150, 200, 300, 500], size=100)

    proteins = [
        {
            'length': int(l),
            'coords': torch.randn(int(l), 3),
            'edge_index': torch.zeros(2, int(l) * 30, dtype=torch.long),
            'edge_features': torch.randn(int(l) * 30, 32)
        }
        for l in lengths
    ]

    print(f"Dataset: {len(proteins)} proteins")
    print(f"Length distribution: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # Naive batching (fixed batch size, pad to global max)
    print("\n1. Naive Batching (fixed batch=8, pad to global max=500):")
    naive_batch_size = 8
    max_len_global = max(lengths)

    naive_tokens = 0
    naive_padding = 0

    for i in range(0, len(proteins), naive_batch_size):
        batch = proteins[i:i + naive_batch_size]
        actual_batch_size = len(batch)
        total = actual_batch_size * max_len_global
        actual_lengths = sum(p['length'] for p in batch)
        padding = total - actual_lengths

        naive_tokens += total
        naive_padding += padding

    naive_efficiency = 1.0 - (naive_padding / naive_tokens)
    print(f"   Total tokens: {naive_tokens}")
    print(f"   Padding tokens: {naive_padding}")
    print(f"   Efficiency: {naive_efficiency:.1%}")

    # Dynamic batching
    print("\n2. Dynamic Batching (length-sorted, adaptive batch size):")

    batcher = LengthBasedBatcher(max_tokens_per_batch=8192)
    buckets = batcher.create_buckets(proteins)

    dynamic_tokens = 0
    dynamic_padding = 0

    for bucket_id, bucket in buckets.items():
        batches = batcher.create_batches(bucket)

        for batch in batches:
            batch_lens = [p['length'] for p in batch]
            max_in_batch = max(batch_lens)
            total = len(batch) * max_in_batch
            actual = sum(batch_lens)
            padding = total - actual

            dynamic_tokens += total
            dynamic_padding += padding

    dynamic_efficiency = 1.0 - (dynamic_padding / dynamic_tokens)
    print(f"   Total tokens: {dynamic_tokens}")
    print(f"   Padding tokens: {dynamic_padding}")
    print(f"   Efficiency: {dynamic_efficiency:.1%}")

    # Comparison
    print("\n" + "="*60)
    print("Improvement:")
    print(f"   Token reduction: {(1 - dynamic_tokens/naive_tokens):.1%}")
    print(f"   Efficiency gain: {dynamic_efficiency/naive_efficiency:.2f}x")
    print(f"   Throughput improvement: {naive_tokens/dynamic_tokens:.2f}x")


if __name__ == "__main__":
    print("Testing Dynamic Batching\n")

    # Test batcher
    batcher = LengthBasedBatcher()

    # Create synthetic dataset
    proteins = [
        {'length': l, 'coords': torch.randn(l, 3)}
        for l in [50, 75, 100, 95, 200, 210, 500, 480]
    ]

    # Create buckets
    buckets = batcher.create_buckets(proteins)

    print("\nBatch sizes by bucket:")
    for bucket_id, bucket in buckets.items():
        if bucket:
            avg_len = np.mean([p['length'] for p in bucket])
            batch_size = batcher.compute_batch_size(int(avg_len))
            print(f"  Bucket {bucket_id} (avg_len={avg_len:.0f}): batch_size={batch_size}")

    # Run comparison
    print("\n")
    compare_batching_strategies()
