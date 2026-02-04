"""
BFloat16 Optimized ProteinMPNN

Optimization: Uses BFloat16 precision for ~2x memory bandwidth improvement.

Key benefits:
- 2x reduction in memory footprint
- 2x improvement in memory bandwidth utilization
- Preserves dynamic range (8-bit exponent like Float32)
- Near-linear speedup for bandwidth-bound autoregressive decoding

Reference: Section 6.1 of optimization document
"""

import torch
import torch.nn as nn
from models.baseline import BaselineProteinMPNN
from typing import Optional


class BFloat16ProteinMPNN(BaselineProteinMPNN):
    """
    BFloat16-optimized version of ProteinMPNN.

    Automatically converts inputs to bfloat16 and runs all operations
    in reduced precision for improved throughput on Apple Silicon M3 Pro.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Convert entire model to BFloat16
        self.to(dtype=torch.bfloat16)

        # Track whether BFloat16 is supported
        self.use_bfloat16 = self._check_bfloat16_support()

        if not self.use_bfloat16:
            print("Warning: BFloat16 not supported on this device, falling back to Float32")
            self.to(dtype=torch.float32)

    def _check_bfloat16_support(self) -> bool:
        """Check if the current device supports BFloat16."""
        try:
            # Try to create a bfloat16 tensor
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
            # Try a simple operation
            _ = test_tensor * 2
            return True
        except (RuntimeError, TypeError):
            return False

    def forward(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Forward pass with automatic BFloat16 conversion.

        Args:
            node_coords: [batch_size, num_nodes, 6]
            edge_index: [2, num_edges]
            edge_distances: [num_edges, 32]
            temperature: Sampling temperature

        Returns:
            sequences: [batch_size, num_nodes] (Int64)
        """
        # Convert inputs to bfloat16 (except integer edge_index)
        if self.use_bfloat16:
            node_coords = node_coords.to(dtype=torch.bfloat16)
            edge_distances = edge_distances.to(dtype=torch.bfloat16)

        # Run parent forward pass
        sequences = super().forward(node_coords, edge_index, edge_distances, temperature)

        return sequences

    def encode_only(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor
    ) -> torch.Tensor:
        """Encode structure in BFloat16."""
        if self.use_bfloat16:
            node_coords = node_coords.to(dtype=torch.bfloat16)
            edge_distances = edge_distances.to(dtype=torch.bfloat16)

        return super().encode_only(node_coords, edge_index, edge_distances)


class MixedPrecisionProteinMPNN(BaselineProteinMPNN):
    """
    Mixed precision variant: encoder in BFloat16, critical ops in Float32.

    Some operations benefit from full precision (e.g., LayerNorm, softmax).
    This variant uses BFloat16 for matmuls and Float32 for numerically sensitive ops.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_bfloat16 = self._check_bfloat16_support()

        if self.use_bfloat16:
            # Convert encoder to BFloat16 (compute-heavy)
            self.encoder.to(dtype=torch.bfloat16)
            # Keep decoder in Float32 for numerical stability in softmax
            self.decoder.to(dtype=torch.float32)
        else:
            print("Warning: BFloat16 not supported, using Float32")

    def _check_bfloat16_support(self) -> bool:
        """Check if the current device supports BFloat16."""
        try:
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
            _ = test_tensor * 2
            return True
        except (RuntimeError, TypeError):
            return False

    def forward(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """Mixed precision forward pass."""
        # Encode in BFloat16
        if self.use_bfloat16:
            node_coords_bf16 = node_coords.to(dtype=torch.bfloat16)
            edge_distances_bf16 = edge_distances.to(dtype=torch.bfloat16)
            encoder_output = self.encoder(node_coords_bf16, edge_index, edge_distances_bf16)

            # Convert encoder output to Float32 for decoder
            encoder_output = encoder_output.to(dtype=torch.float32)
        else:
            encoder_output = self.encoder(node_coords, edge_index, edge_distances)

        # Add batch dimension if needed
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(0)

        # Decode in Float32
        sequences = self.decoder.generate_sequence(encoder_output, temperature)

        return sequences


# Utility function to convert existing model to BFloat16
def convert_to_bfloat16(model: nn.Module, inplace: bool = True) -> Optional[nn.Module]:
    """
    Convert a trained model to BFloat16 precision.

    Args:
        model: PyTorch model
        inplace: Whether to modify the model in-place

    Returns:
        Converted model (if not inplace) or None
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    # Check support
    try:
        test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
        _ = test_tensor * 2
    except (RuntimeError, TypeError):
        print("BFloat16 not supported on this device")
        return model if not inplace else None

    # Convert model
    model.to(dtype=torch.bfloat16)

    print(f"Model converted to BFloat16")
    print(f"Expected memory reduction: ~2x")
    print(f"Expected speedup on bandwidth-bound ops: 1.5x-2x")

    return model if not inplace else None


if __name__ == "__main__":
    # Test BFloat16 model
    print("Testing BFloat16 ProteinMPNN...")

    model = BFloat16ProteinMPNN(
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3
    )

    # Create dummy input
    batch_size = 2
    seq_len = 50
    num_edges = seq_len * 30  # k=30 neighbors

    node_coords = torch.randn(batch_size, seq_len, 6)
    edge_index = torch.randint(0, seq_len, (2, num_edges))
    edge_distances = torch.randn(num_edges, 32)

    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Input shape: {node_coords.shape}")

    # Run forward pass
    try:
        sequences = model(node_coords, edge_index, edge_distances)
        print(f"Output shape: {sequences.shape}")
        print("BFloat16 model test passed!")
    except Exception as e:
        print(f"Error: {e}")
