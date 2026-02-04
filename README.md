# ProteinMPNN_apx: Validated MLX Implementation

**Real benchmarks of ProteinMPNN optimizations on Apple Silicon**

## ✅ VALIDATION STATUS: REAL MEASUREMENTS ONLY

**All results in this repository are from ACTUAL benchmarking runs.**

- Device: Apple Silicon with MLX 0.30.3
- Measurement method: `time.perf_counter()` with proper warmup
- Runs per test: 50 iterations with 10 warmup runs
- Validation date: 2026-02-04

---

## Overview

This repository contains **validated implementations** of ProteinMPNN optimizations that can actually run on Apple Silicon using the MLX framework.

### What Was Removed

**During validation, the following were removed because they could not be tested:**
- ❌ All PyTorch-based implementations (PyTorch not installed)
- ❌ All ONNX Runtime implementations (ONNX Runtime not installed)
- ❌ All simulated benchmark results
- ❌ Unverified performance claims

### What Remains

**Only validated, working implementations:**
- ✅ MLX Baseline implementation
- ✅ MLX FP16 variant
- ✅ MLX Optimized variant (reduced complexity)
- ✅ Real benchmark measurements
- ✅ Actual timing data with confidence intervals

---

## Real Performance Results

### Actual Measurements (Not Simulated)

All numbers below are from real benchmark runs on Apple Silicon with MLX 0.30.3.

| Variant | 50-res | 100-res | 200-res | Avg Speedup |
|---------|--------|---------|---------|-------------|
| **Baseline MLX** | 143.5 res/sec | 127.9 res/sec | 129.3 res/sec | 1.00x (reference) |
| **FP16 MLX** | 138.2 res/sec | 137.8 res/sec | 131.9 res/sec | **1.02x** |
| **Optimized MLX** | 259.1 res/sec | 266.7 res/sec | 271.7 res/sec | **2.00x** |

### Key Findings

1. **FP16 provides minimal improvement**: 0.96-1.08x speedup
   - Sometimes slightly slower due to conversion overhead
   - MLX may already optimize precision internally
   - Not worth the complexity for this use case

2. **Model reduction is effective**: 1.81-2.10x speedup
   - Reducing layers (2→1) and hidden size (64→32)
   - Maintains reasonable throughput
   - Trade-off: reduced model capacity

3. **Absolute performance**: 130-270 res/sec
   - Suitable for interactive protein design
   - 100-residue protein: 0.38-0.78 seconds
   - Scales roughly linearly with sequence length

---

## Implementation Details

### Baseline MLX Implementation

**Architecture:**
- Hidden dimension: 64
- Number of MPNN layers: 2
- k-nearest neighbors: 30
- Features: 16-dimensional positional encoding
- Edge features: 16-dimensional RBF encoding of distances

**Real Feature Extraction:**
```python
# Actual RBF encoding (not placeholder)
def rbf_encode(distances, d_min=0.0, d_max=20.0, d_count=16):
    d_mu = mx.linspace(d_min, d_max, d_count)
    d_sigma = (d_max - d_min) / d_count
    return mx.exp(-((distances - d_mu) ** 2) / (2 * d_sigma ** 2))

# Actual k-NN graph from coordinates
def build_knn_graph(coords, k=30):
    dist_matrix = compute_distances(coords)  # O(N²)
    nearest_indices = mx.argsort(dist_matrix)[:, :k]
    return edge_index, distances

# Real message passing
class MLXMPNNLayer:
    def __call__(self, node_h, edge_index, edge_features):
        # Gather source and destination
        src_h = node_h[edge_index[0]]
        dst_h = node_h[edge_index[1]]

        # Compute messages
        messages = self.w_msg(concat([src_h, dst_h, edge_features]))

        # Aggregate with scatter_add
        aggregated = scatter_add(messages, edge_index[1])

        # Update nodes
        return node_h + self.w_update(concat([node_h, aggregated]))
```

### Optimized MLX Implementation

**Changes from baseline:**
- Hidden dimension: 64 → 32 (2x reduction)
- Number of layers: 2 → 1 (2x reduction)
- Result: ~2x speedup

---

## Benchmark Methodology

### Proper Timing Protocol

```python
# Warmup (critical for MLX)
for _ in range(warmup_runs):
    logits = model(coords)
    mx.eval(logits)  # Force evaluation

# Actual measurement
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    logits = model(coords)
    mx.eval(logits)  # Force evaluation
    end = time.perf_counter()
    times.append(end - start)

# Statistics
mean_time = np.mean(times)
std_time = np.std(times)
```

### Why This Method Is Correct

1. **Warmup runs**: MLX compiles graphs on first execution
2. **Force evaluation**: `mx.eval()` ensures computation completes
3. **Multiple runs**: Statistical confidence (mean ± std)
4. **High-resolution timing**: `time.perf_counter()` for precision

---

## Installation

```bash
# MLX (Apple Silicon only)
pip install mlx

# NumPy for numerical operations
pip install numpy
```

**System Requirements:**
- macOS with Apple Silicon (M1, M2, M3, M4)
- Python 3.8+
- 8GB+ RAM recommended

---

## Usage

### Running Benchmarks

```bash
# Run complete benchmark suite
python3 run_real_benchmarks.py

# Results saved to: output/benchmarks/real_measurements.json
```

### Using the Models

```python
import mlx.core as mx
from run_real_benchmarks import RealMLXProteinMPNN, create_test_protein

# Create model
model = RealMLXProteinMPNN(hidden_dim=64, num_layers=2)

# Create test protein (100 residues)
coords = create_test_protein(length=100)

# Forward pass
logits = model(coords, k=30)
mx.eval(logits)

# Sample sequence
sequence = mx.argmax(logits, axis=-1)
print(f"Predicted sequence: {sequence}")
```

---

## Detailed Results

### Timing Statistics (100-residue protein)

**Baseline MLX:**
- Mean: 781.87 ± 101.81 ms
- Median: 745.27 ms
- Min: 698.44 ms
- Throughput: 127.9 res/sec

**FP16 MLX:**
- Mean: 725.85 ± 26.22 ms
- Median: 721.07 ms
- Min: 692.21 ms
- Throughput: 137.8 res/sec
- **Speedup: 1.08x**

**Optimized MLX:**
- Mean: 375.01 ± 34.22 ms
- Median: 363.55 ms
- Min: 337.11 ms
- Throughput: 266.7 res/sec
- **Speedup: 2.08x**

---

## Project Structure

```
ProteinMPNN_apx/
├── README.md                          # This file (REAL results only)
├── run_real_benchmarks.py             # Complete benchmark script
├── models/
│   ├── reference_implementation.py    # Reference with full features
│   └── README.md                      # Model documentation
├── output/
│   └── benchmarks/
│       └── real_measurements.json     # ACTUAL benchmark data
└── TRANSPARENCY_REPORT.md             # Validation methodology
```

---

## Limitations and Future Work

### Current Limitations

1. **Simplified Model**: This is not full ProteinMPNN
   - Missing: Amino acid type features
   - Missing: Complete encoder-decoder architecture
   - Missing: Autoregressive sampling loop
   - Present: Core MPNN structure and optimization patterns

2. **Single Framework**: MLX only
   - PyTorch implementations removed (couldn't validate)
   - ONNX implementations removed (couldn't validate)
   - Limits portability but ensures honesty

3. **Limited Sequence Lengths**: Tested up to 200 residues
   - Longer sequences would require more memory
   - Linear scaling suggests 1000-residue feasible
   - Not validated beyond 200

### Future Improvements

**If you have PyTorch/CUDA available:**
- Implement and validate PyTorch MPS backend
- Test Flash Attention for longer sequences
- Compare with CUDA baseline on discrete GPU
- Validate accuracy against official ProteinMPNN

**If you want better performance:**
- Profile with MLX instruments
- Optimize graph construction (currently O(N²))
- Implement kernel fusion for message passing
- Add batching support

**If you want complete ProteinMPNN:**
- Add amino acid type embeddings
- Implement full encoder-decoder
- Add autoregressive sampling
- Validate against official checkpoint

---

## Validation Data

**Complete benchmark results**: [output/benchmarks/real_measurements.json](output/benchmarks/real_measurements.json)

**Validation details:**
- Timestamp: 2026-02-04T10:55:37
- MLX version: 0.30.3
- NumPy version: 2.4.2
- Device: Apple Silicon
- Test lengths: 50, 100, 200 residues
- Runs per test: 50
- Warmup runs: 10

---

## Comparison with Literature

**Reported speedups in literature:**
- Flash Attention: 2-4x for long sequences
- KV Caching: 5-10x for autoregressive
- FP16: 1.5-2x with minimal accuracy loss
- Combined: 10-25x claimed

**Our actual results:**
- FP16: 1.02x (minimal improvement)
- Optimized: 2.08x (model reduction)
- Combined: Not tested (no PyTorch)

**Why the difference?**
1. Literature uses discrete GPUs (different architecture)
2. Literature uses complete ProteinMPNN (more optimization opportunities)
3. Literature may include preprocessing in baseline (we measure pure inference)
4. MLX may already optimize internally (negating some techniques)

---

## Conclusion

**What this repository provides:**
- ✅ Real, validated benchmark data
- ✅ Working MLX implementations
- ✅ Honest assessment of performance
- ✅ Proper benchmarking methodology

**What this repository does NOT provide:**
- ❌ PyTorch implementations (couldn't validate)
- ❌ Spectacular speedups (2x is realistic, not 20x)
- ❌ Complete ProteinMPNN (simplified for validation)
- ❌ Production-ready code (research prototype)

**Use this as:**
- Reference for MLX optimization patterns
- Starting point for your own implementations
- Reality check on optimization claims
- Example of proper benchmarking

---

**For questions or contributions:** Open an issue on GitHub

**Last updated:** 2026-02-04
**Validation status:** ✅ All results verified
