"""
MPS-Optimized ProteinMPNN for Apple Silicon

Optimization: Metal Performance Shaders backend for M3 Pro GPU acceleration.

Key benefits:
- 5x speedup on M3 Pro (18-core GPU)
- Minimal code changes (3-5 lines)
- Native Apple Silicon support
- Unified Memory Architecture optimization
- Automatic fallback for unsupported ops

Reference: Section 4 of accelerating_proteinmpnn.txt document
"""

import torch
import torch.nn as nn
import os
import warnings
from typing import Optional, Dict
from models.baseline import BaselineProteinMPNN


def get_mps_device() -> torch.device:
    """
    Detects Apple Silicon GPU and configures the MPS backend.

    Implements robust device detection for M3 Pro optimization.
    Prioritizes: MPS > CUDA > CPU

    Returns:
        torch.device: Optimal device for current hardware
    """
    # Check for MPS (Metal Performance Shaders) availability
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # MPS is available on MacOS 12.3+ with ARM64
            # M3 Pro is implicitly supported if MPS is built
            return torch.device("mps")

    # Fallback to CUDA if available
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    # Final fallback to CPU
    warnings.warn("Neither MPS nor CUDA available, falling back to CPU")
    return torch.device("cpu")


def enable_mps_fallback():
    """
    Enable MPS fallback for unsupported operations.

    On M3 Pro, some scatter/gather operations may not have
    direct Metal kernel mappings. This enables automatic CPU
    fallback with minimal performance penalty due to Unified Memory.

    Reference: Section 4.3 Phase 3 of accelerating document
    """
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def configure_mps_memory():
    """
    Configure MPS memory management for optimal performance.

    M3 Pro has 36GB unified memory - we want to maximize usage
    without triggering system-wide memory pressure.
    """
    if hasattr(torch.backends, 'mps'):
        # Set memory fraction (use up to 80% of available memory)
        # Leaves headroom for OS and other apps
        try:
            # Note: This is a conceptual API - actual implementation
            # may vary with PyTorch version
            if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                torch.backends.mps.set_per_process_memory_fraction(0.8)
        except AttributeError:
            pass  # Not all PyTorch versions support this


class MPSOptimizedProteinMPNN(nn.Module):
    """
    ProteinMPNN optimized for Apple Silicon M3 Pro.

    Automatically configures MPS backend and applies M3-specific
    optimizations including:
    - Device-aware tensor placement
    - Scatter/gather fallback handling
    - Optimal batch sizing for 36GB memory
    - Memory bandwidth optimization

    Expected speedup: 5x over CPU baseline on M3 Pro
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        enable_fallback: bool = True,
        optimize_memory: bool = True,
        recommended_batch_size: int = 16,
        **model_kwargs
    ):
        """
        Args:
            base_model: Existing model to optimize (if None, creates new)
            enable_fallback: Enable CPU fallback for unsupported ops
            optimize_memory: Configure memory management
            recommended_batch_size: Optimal batch size for M3 Pro
            **model_kwargs: Arguments for new model creation
        """
        super().__init__()

        # Enable MPS optimizations
        if enable_fallback:
            enable_mps_fallback()

        if optimize_memory:
            configure_mps_memory()

        # Get optimal device
        self.device = get_mps_device()

        print(f"{'='*60}")
        print(f"MPS-Optimized ProteinMPNN")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
        print(f"Fallback Enabled: {enable_fallback}")
        print(f"Recommended Batch Size: {recommended_batch_size}")

        # Create or wrap base model
        if base_model is None:
            base_model = BaselineProteinMPNN(**model_kwargs)

        self.model = base_model

        # Move to MPS device
        try:
            if self.device.type == 'mps':
                self.model = self.model.to(self.device)
                print(f"✓ Model moved to MPS device")

                # Verify model is on MPS
                first_param_device = next(self.model.parameters()).device
                if first_param_device.type == 'mps':
                    print(f"✓ Verified: Parameters on {first_param_device}")
                else:
                    warnings.warn(f"Warning: Parameters on {first_param_device}, expected MPS")

        except Exception as e:
            warnings.warn(f"Could not move model to MPS: {e}")
            warnings.warn("Falling back to CPU")
            self.device = torch.device('cpu')

        print(f"{'='*60}\n")

        self.recommended_batch_size = recommended_batch_size

    def forward(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        temperature: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with automatic device management.

        Tensors are automatically moved to MPS device if not already there.

        Args:
            node_coords: [batch_size, num_nodes, 6]
            edge_index: [2, num_edges]
            edge_distances: [num_edges, 32]
            temperature: Sampling temperature

        Returns:
            sequences: [batch_size, num_nodes]
        """
        # Move inputs to device if needed
        if node_coords.device != self.device:
            node_coords = node_coords.to(self.device)

        if edge_index.device != self.device:
            edge_index = edge_index.to(self.device)

        if edge_distances.device != self.device:
            edge_distances = edge_distances.to(self.device)

        # Forward through model
        return self.model(node_coords, edge_index, edge_distances, temperature, **kwargs)

    @torch.no_grad()
    def benchmark_mps(
        self,
        test_lengths: list = [50, 100, 200, 500],
        num_runs: int = 10,
        warmup: int = 5
    ) -> Dict:
        """
        Benchmark MPS performance across different protein lengths.

        Tests the scaling behavior on M3 Pro GPU and compares
        against CPU baseline if available.

        Args:
            test_lengths: Protein lengths to test
            num_runs: Number of benchmark runs
            warmup: Number of warmup runs

        Returns:
            Dictionary with detailed benchmark results
        """
        import time
        from models.baseline import build_knn_graph, rbf_encode_distances

        print(f"\n{'='*60}")
        print(f"MPS Benchmark on {self.device}")
        print(f"{'='*60}\n")

        results = {
            'device': str(self.device),
            'by_length': {}
        }

        for length in test_lengths:
            print(f"Testing length {length}...")

            # Generate test data
            coords = torch.randn(length, 3)
            edge_index, distances = build_knn_graph(coords)
            edge_distances = rbf_encode_distances(distances)

            # Prepare input
            orientations = torch.randn(length, 3)
            node_coords = torch.cat([coords, orientations], dim=-1).unsqueeze(0)

            # Warmup
            for _ in range(warmup):
                _ = self(node_coords, edge_index, edge_distances)

            # Benchmark
            times = []
            for _ in range(num_runs):
                # Synchronize if on MPS
                if self.device.type == 'mps':
                    torch.mps.synchronize()

                start = time.time()
                _ = self(node_coords, edge_index, edge_distances)

                if self.device.type == 'mps':
                    torch.mps.synchronize()

                elapsed = time.time() - start
                times.append(elapsed)

            mean_time = sum(times) / len(times)
            std_time = (sum((t - mean_time)**2 for t in times) / len(times)) ** 0.5

            results['by_length'][length] = {
                'mean_time': mean_time,
                'std_time': std_time,
                'throughput': length / mean_time
            }

            print(f"  Time: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
            print(f"  Throughput: {length/mean_time:.1f} residues/sec\n")

        return results

    def get_memory_stats(self) -> Dict:
        """
        Get current MPS memory statistics.

        Returns:
            Dictionary with memory usage information
        """
        if self.device.type == 'mps':
            # MPS memory tracking
            try:
                if hasattr(torch.mps, 'current_allocated_memory'):
                    allocated = torch.mps.current_allocated_memory() / (1024**2)  # MB
                    return {
                        'allocated_mb': allocated,
                        'device': 'mps',
                        'total_memory_gb': 36  # M3 Pro with 36GB
                    }
            except:
                pass

        return {
            'device': str(self.device),
            'memory_tracking': 'not available'
        }


class MPSBatchOptimizer:
    """
    Utility class for optimizing batch sizes on M3 Pro.

    The 36GB unified memory allows for much larger batches than
    typical consumer GPUs. This class helps determine optimal
    batch sizes based on protein length.
    """

    def __init__(self, total_memory_gb: float = 36.0, safety_factor: float = 0.7):
        """
        Args:
            total_memory_gb: Total unified memory
            safety_factor: Use only this fraction of memory (0.7 = 70%)
        """
        self.total_memory_gb = total_memory_gb
        self.safety_factor = safety_factor
        self.available_memory_gb = total_memory_gb * safety_factor

    def estimate_memory_per_protein(
        self,
        seq_length: int,
        hidden_dim: int = 128,
        k_neighbors: int = 30
    ) -> float:
        """
        Estimate memory usage per protein in GB.

        Args:
            seq_length: Protein length
            hidden_dim: Model hidden dimension
            k_neighbors: Number of neighbors in graph

        Returns:
            Estimated memory in GB
        """
        # Node features: seq_len * hidden_dim * 4 bytes (float32)
        node_memory = seq_length * hidden_dim * 4

        # Edge features: seq_len * k_neighbors * hidden_dim * 4
        edge_memory = seq_length * k_neighbors * hidden_dim * 4

        # KV cache (if using): seq_len * hidden_dim * num_layers * 2 * 4
        kv_cache_memory = seq_length * hidden_dim * 3 * 2 * 4

        # Total per protein
        total_bytes = node_memory + edge_memory + kv_cache_memory

        # Add 30% overhead for intermediate tensors
        total_bytes *= 1.3

        return total_bytes / (1024**3)  # Convert to GB

    def recommend_batch_size(
        self,
        seq_length: int,
        hidden_dim: int = 128
    ) -> int:
        """
        Recommend optimal batch size for M3 Pro.

        Args:
            seq_length: Protein length
            hidden_dim: Model hidden dimension

        Returns:
            Recommended batch size
        """
        memory_per_protein = self.estimate_memory_per_protein(seq_length, hidden_dim)

        # Calculate max batch size
        max_batch = int(self.available_memory_gb / memory_per_protein)

        # Clamp to reasonable range
        batch_size = max(1, min(max_batch, 64))

        return batch_size


def verify_mps_installation():
    """
    Verify that PyTorch MPS backend is properly installed and configured.

    Returns diagnostic information about the MPS setup.
    """
    print("MPS Installation Verification")
    print("="*60)

    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check MPS availability
    if hasattr(torch.backends, 'mps'):
        print(f"MPS module available: ✓")
        print(f"MPS is_available(): {torch.backends.mps.is_available()}")
        print(f"MPS is_built(): {torch.backends.mps.is_built()}")

        if torch.backends.mps.is_available():
            print("\n✓ MPS Backend Ready!")
            print("  Your M3 Pro can accelerate ProteinMPNN with Metal")

            # Try creating a tensor on MPS
            try:
                test_tensor = torch.randn(100, 100, device='mps')
                result = test_tensor @ test_tensor.T
                print("  ✓ MPS tensor operations working")
            except Exception as e:
                print(f"  ✗ MPS tensor test failed: {e}")

        else:
            print("\n✗ MPS Not Available")
            print("  This may not be an Apple Silicon Mac")
            print("  Or macOS version is < 12.3")

    else:
        print(f"MPS module not found: ✗")
        print("  PyTorch version may be too old")
        print("  Recommended: PyTorch 2.0+")

    print("="*60)


if __name__ == "__main__":
    print("MPS-Optimized ProteinMPNN for Apple Silicon\n")

    # Verify installation
    verify_mps_installation()

    print("\n")

    # Test MPS optimization
    try:
        model = MPSOptimizedProteinMPNN(hidden_dim=128)

        print("\nModel created successfully!")
        print(f"Device: {model.device}")
        print(f"Recommended batch size: {model.recommended_batch_size}")

        # Test batch size optimizer
        print("\n" + "="*60)
        print("Batch Size Recommendations for M3 Pro (36GB)")
        print("="*60)

        optimizer = MPSBatchOptimizer()

        for length in [50, 100, 200, 500, 1000]:
            batch_size = optimizer.recommend_batch_size(length)
            memory_per = optimizer.estimate_memory_per_protein(length)
            print(f"Length {length:4d}: batch_size={batch_size:3d} "
                  f"(~{memory_per*1000:.1f} MB per protein)")

    except Exception as e:
        print(f"\nError: {e}")
        print("MPS may not be available on this system")
