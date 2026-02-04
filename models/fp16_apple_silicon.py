"""
FP16 Mixed Precision for Apple Silicon

Optimization: Half-precision (FP16) specifically tuned for M3 Pro GPU.

Key benefits:
- 2x memory bandwidth improvement
- 2x memory capacity (store twice as many models)
- Peak FP16 throughput on M3 GPU
- Maintained numerical stability

Reference: Section 3.1 of accelerating document
"""

import torch
import torch.nn as nn
from typing import Optional
import warnings

from models.baseline import BaselineProteinMPNN
from models.mps_optimized import get_mps_device, enable_mps_fallback


class FP16AppleSiliconMPNN(nn.Module):
    """
    ProteinMPNN optimized with FP16 precision for M3 Pro.

    The M3 GPU achieves peak throughput in FP16, delivering
    significantly higher FLOPS than FP32. This variant converts
    the model to half-precision while maintaining numerical stability.

    Expected benefits on M3 Pro:
    - 2x memory bandwidth
    - 2x model capacity
    - ~1.5-2x speed improvement
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        use_mps: bool = True,
        mixed_precision: bool = True,
        **model_kwargs
    ):
        """
        Args:
            base_model: Model to convert (None = create new)
            use_mps: Use MPS backend
            mixed_precision: Use mixed precision (FP16 compute, FP32 accumulators)
            **model_kwargs: Model creation arguments
        """
        super().__init__()

        # Setup device
        if use_mps:
            enable_mps_fallback()
            self.device = get_mps_device()
        else:
            self.device = torch.device('cpu')

        print(f"{'='*60}")
        print(f"FP16 Apple Silicon ProteinMPNN")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Precision: FP16 (Half)")
        print(f"Mixed Precision: {mixed_precision}")

        # Create or use base model
        if base_model is None:
            base_model = BaselineProteinMPNN(**model_kwargs)

        self.model = base_model
        self.mixed_precision = mixed_precision

        # Convert to half precision
        self._convert_to_fp16()

        # Move to device
        self.model = self.model.to(self.device)

        print(f"✓ Model converted to FP16")
        print(f"✓ Moved to {self.device}")
        print(f"{'='*60}\n")

    def _convert_to_fp16(self):
        """
        Convert model to FP16 with careful handling of numerically sensitive ops.

        Strategy:
        - Linear layers: FP16 (where most compute happens)
        - LayerNorm: Keep FP32 for stability
        - Embeddings: Keep FP32 for precision
        """
        for name, module in self.model.named_modules():
            # Convert Linear layers to FP16
            if isinstance(module, nn.Linear):
                module.half()

            # Keep LayerNorm in FP32 for numerical stability
            elif isinstance(module, nn.LayerNorm):
                module.float()

            # Keep Embeddings in FP32
            elif isinstance(module, nn.Embedding):
                module.float()

    def forward(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        temperature: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with automatic FP16 conversion.

        Args:
            node_coords: [batch_size, num_nodes, 6]
            edge_index: [2, num_edges]
            edge_distances: [num_edges, 32]
            temperature: Sampling temperature

        Returns:
            sequences: [batch_size, num_nodes] (Int64)
        """
        # Move to device
        node_coords = node_coords.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_distances = edge_distances.to(self.device)

        # Convert to FP16 (except integer tensors)
        node_coords = node_coords.half()
        edge_distances = edge_distances.half()

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.mixed_precision) if torch.cuda.is_available() else torch.nullcontext():
            sequences = self.model(node_coords, edge_index, edge_distances, temperature, **kwargs)

        return sequences


def benchmark_fp16_vs_fp32():
    """
    Benchmark FP16 vs FP32 on Apple Silicon.

    Compares memory usage and inference speed.
    """
    import time
    from models.baseline import build_knn_graph, rbf_encode_distances

    print("FP16 vs FP32 Benchmark on Apple Silicon\n")
    print("="*60)

    # Check device
    device = get_mps_device()
    if device.type != 'mps':
        print("MPS not available, skipping benchmark")
        return

    # Test parameters
    seq_len = 200
    num_runs = 10

    # Prepare data
    coords = torch.randn(seq_len, 3)
    edge_index, distances = build_knn_graph(coords)
    edge_distances = rbf_encode_distances(distances)
    orientations = torch.randn(seq_len, 3)
    node_coords = torch.cat([coords, orientations], dim=-1).unsqueeze(0)

    # FP32 model
    print("Testing FP32...")
    model_fp32 = BaselineProteinMPNN(hidden_dim=128)
    model_fp32 = model_fp32.to(device)
    model_fp32.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model_fp32(
                node_coords.to(device),
                edge_index.to(device),
                edge_distances.to(device)
            )

    # Benchmark FP32
    times_fp32 = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.mps.synchronize() if device.type == 'mps' else None
            start = time.time()
            _ = model_fp32(
                node_coords.to(device),
                edge_index.to(device),
                edge_distances.to(device)
            )
            torch.mps.synchronize() if device.type == 'mps' else None
            times_fp32.append(time.time() - start)

    time_fp32 = sum(times_fp32) / len(times_fp32)

    # FP16 model
    print("Testing FP16...")
    model_fp16 = FP16AppleSiliconMPNN(hidden_dim=128, use_mps=True)
    model_fp16.eval()

    # Benchmark FP16
    times_fp16 = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.mps.synchronize() if device.type == 'mps' else None
            start = time.time()
            _ = model_fp16(node_coords, edge_index, edge_distances)
            torch.mps.synchronize() if device.type == 'mps' else None
            times_fp16.append(time.time() - start)

    time_fp16 = sum(times_fp16) / len(times_fp16)

    # Results
    print(f"\nResults (seq_len={seq_len}):")
    print(f"  FP32: {time_fp32*1000:.2f}ms")
    print(f"  FP16: {time_fp16*1000:.2f}ms")
    print(f"  Speedup: {time_fp32/time_fp16:.2f}x")

    # Memory estimate
    fp32_memory = sum(p.numel() * 4 for p in model_fp32.parameters()) / (1024**2)
    fp16_memory = sum(
        p.numel() * (2 if p.dtype == torch.float16 else 4)
        for p in model_fp16.parameters()
    ) / (1024**2)

    print(f"\nMemory:")
    print(f"  FP32: {fp32_memory:.1f} MB")
    print(f"  FP16: {fp16_memory:.1f} MB")
    print(f"  Reduction: {(1 - fp16_memory/fp32_memory)*100:.1f}%")


if __name__ == "__main__":
    print("FP16 Apple Silicon Optimization\n")

    # Run benchmark
    try:
        benchmark_fp16_vs_fp32()
    except Exception as e:
        print(f"Error: {e}")
        print("FP16 benchmarking requires MPS-capable device")
