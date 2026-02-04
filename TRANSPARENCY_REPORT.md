# Transparency Report: Benchmark Results and Implementation Status

**Date**: 2026-02-04
**Version**: All versions (v0.1.0 - v0.5.0)

---

## ⚠️ CRITICAL DISCLAIMER

This document provides complete transparency about the **nature and authenticity** of benchmark results in this repository. **All users must read this before using any reported performance metrics.**

---

## Executive Summary

**IMPORTANT**: The benchmark results reported in this repository are **SIMULATED ESTIMATES** based on theoretical speedup factors from the reference literature, **NOT actual timing measurements** from running the implementations.

### What This Means

- ✅ **Model implementations**: Functional code that demonstrates optimization techniques
- ✅ **Architectural patterns**: Correct implementation of optimization strategies
- ❌ **Benchmark numbers**: Theoretical estimates, NOT measured on actual hardware
- ❌ **Throughput metrics**: Calculated from baseline assumptions, NOT empirical
- ❌ **Memory usage**: Estimated based on tensor sizes, NOT measured runtime profiling

---

## Detailed Analysis by Component

### 1. Model Implementations (models/*.py)

#### Status: **PARTIAL** - Functional but Incomplete

**What's Implemented**:
- ✅ Correct architectural patterns for each optimization
- ✅ Proper PyTorch/MLX module structure
- ✅ Optimization-specific layers (Flash Attention, KV Cache, etc.)
- ✅ Device selection logic (MPS, CUDA, CPU)
- ✅ Precision conversion (FP16, BFloat16)

**What's Missing/Placeholder**:
- ❌ **Actual protein feature extraction**: Models use `torch.randn()` for features instead of:
  - RBF encoding of distances
  - Positional encodings
  - Residue type embeddings
  - Orientation features

- ❌ **Real graph construction**: Edge indices are assumed, not built from coordinates

- ❌ **Complete forward passes**: Simplified architectures missing:
  - Multiple MPNN layers with proper message passing
  - Full autoregressive decoding loop
  - Proper masking and attention patterns

**Example from models/baseline.py**:
```python
# Current (Placeholder):
features = torch.randn(B, N, 128, device=coords.device)

# Should be (Real):
features = self.encode_geometry(coords, distances)  # RBF, angles, dihedrals
```

#### Verdict: **Models are architectural demonstrations, not production-ready implementations**

---

### 2. Benchmark Results (output/benchmarks/*.json)

#### Status: **SIMULATED** - Not from Actual Measurements

**How Numbers Were Generated**:

All benchmark JSON files contain estimates calculated as:

```python
# Pseudo-code for simulation
baseline_time = sequence_length / 40.8  # Assumed baseline throughput
speedup_factor = {
    'bfloat16': 1.8,
    'kv_cached': 5.9,
    'flash_attention': 2.0,
    # ... etc
}
optimized_time = baseline_time / speedup_factor
throughput = sequence_length / optimized_time
```

**Speedup Factors Source**:
- Derived from reference literature (long_proteinmpnn.txt)
- Based on theoretical complexity analysis (O(N²) → O(N))
- Extrapolated from published papers on similar architectures

**What Was NOT Done**:
- ❌ No actual `time.perf_counter()` measurements
- ❌ No GPU profiling with CUDA/Metal tools
- ❌ No memory profiling with `tracemalloc` or `torch.cuda.memory_allocated()`
- ❌ No repeated runs for statistical confidence intervals
- ❌ No validation on actual PDB structures

#### Files Affected:
- `simulated_results.json` (v0.1.0)
- `comprehensive_results.json` (v0.2.0)
- `apple_silicon_results.json` (v0.3.0)
- `advanced_optimizations_results.json` (v0.4.0)
- `ultimate_combinations_results.json` (v0.5.0)

#### Verdict: **All benchmark numbers are theoretical estimates**

---

### 3. Accuracy Claims

#### Status: **LITERATURE-BASED** - Not Empirically Validated

**Accuracy Metrics Reported**:
- BFloat16: "<0.5% accuracy loss"
- Int8 Quantization: "<1% accuracy loss"
- Flash Attention: "Mathematically equivalent, 0% loss"

**Source**:
- Cited from reference documents
- Based on published research on these techniques
- Assumed to transfer to ProteinMPNN architecture

**What Was NOT Done**:
- ❌ No sequence recovery rate measurements
- ❌ No comparison with ground truth ProteinMPNN outputs
- ❌ No validation on benchmark protein datasets
- ❌ No perplexity or cross-entropy measurements

#### Verdict: **Accuracy claims are literature-derived assumptions**

---

### 4. Memory Usage Estimates

#### Status: **CALCULATED** - Tensor Size Math, Not Measured

**Estimation Method**:
```python
# Example calculation
hidden_dim = 128
num_heads = 8
seq_length = 1000

# Standard attention
attention_mem = seq_length * seq_length * num_heads * 4 / 1e6  # MB

# Flash attention
flash_mem = seq_length * block_size * num_heads * 4 / 1e6  # MB
```

**Assumptions**:
- FP32 = 4 bytes, FP16 = 2 bytes
- Ignores framework overhead (PyTorch allocator, MPS metal buffers)
- Assumes perfect packing, no fragmentation
- Doesn't account for gradient buffers, optimizer states

**What Was NOT Done**:
- ❌ No actual memory profiling
- ❌ No peak memory measurements
- ❌ No OOM testing to find actual limits

#### Verdict: **Memory estimates are lower bounds from tensor arithmetic**

---

## Why This Matters

### For Research Use:
- **Do NOT cite** the specific speedup numbers (22.47x, etc.) without validation
- **Do cite** the optimization techniques and architectural patterns
- **Do validate** on your own hardware before publication

### For Production Use:
- **Do NOT deploy** without thorough testing
- **Do expect** different performance characteristics
- **Do profile** on your actual workload and hardware

### For Educational Use:
- **Do use** as a reference for optimization strategies
- **Do understand** the theoretical foundations
- **Do NOT assume** the numbers transfer directly

---

## What Would Be Required for Real Benchmarks

### Minimum Requirements:

1. **Complete Model Implementation**:
   - Full protein feature extraction from PDB files
   - Proper graph construction (k-NN, RBF encoding)
   - Complete encoder-decoder with all layers
   - Autoregressive sampling loop

2. **Actual Timing Infrastructure**:
   ```python
   import time
   times = []
   for _ in range(100):  # Repeated runs
       torch.mps.synchronize()  # Ensure GPU completion
       start = time.perf_counter()
       output = model(protein_batch)
       torch.mps.synchronize()
       end = time.perf_counter()
       times.append(end - start)

   mean_time = np.mean(times)
   std_time = np.std(times)
   ```

3. **Memory Profiling**:
   ```python
   import tracemalloc
   tracemalloc.start()
   output = model(protein_batch)
   current, peak = tracemalloc.get_traced_memory()
   ```

4. **Accuracy Validation**:
   - Load official ProteinMPNN checkpoints
   - Run on CASP14 or similar benchmark
   - Compare sequence recovery rates
   - Compute perplexity on validation set

5. **Hardware Diversity**:
   - Test on actual M3 Pro 36GB
   - Test on M3 Max (for comparison)
   - Test on discrete NVIDIA GPUs (for baseline)
   - Document exact system specs (macOS version, PyTorch version)

---

## Recommendations for Users

### If You Need Real Benchmarks:

1. **Start with small-scale validation**:
   - Run existing model on 1-2 proteins
   - Measure actual wall-clock time
   - Compare with CPU baseline

2. **Use established frameworks**:
   - Download official ProteinMPNN from Baker Lab
   - Apply optimizations incrementally
   - Validate each step maintains accuracy

3. **Profile your specific use case**:
   - Your protein lengths
   - Your batch sizes
   - Your hardware configuration

### If You Want to Contribute:

We welcome contributions that add:
- ✅ Real benchmark measurements on M3 Pro
- ✅ Complete protein feature extraction
- ✅ Validation against official ProteinMPNN
- ✅ Memory profiling data
- ✅ Accuracy metrics on benchmark datasets

---

## Attribution and Sources

### What's Original:
- Implementation architecture and code organization
- Combination of optimization techniques
- Documentation structure

### What's Derived:
- Optimization strategies (from literature)
- Speedup factors (from reference documents)
- Theoretical complexity analysis (from papers)

### Primary References:
1. **long_proteinmpnn.txt** - Comprehensive technical analysis
2. **ProteinMPNN Paper** (Dauparas et al.) - Original architecture
3. **Flash Attention Paper** (Dao et al.) - Attention optimization
4. **PyTorch Documentation** - MPS backend implementation

---

## Conclusion

This repository serves as:
- ✅ **Educational resource** for understanding optimization techniques
- ✅ **Architectural reference** for implementing optimizations
- ✅ **Theoretical analysis** of performance potential

This repository DOES NOT serve as:
- ❌ **Production-ready implementation** of ProteinMPNN
- ❌ **Empirical benchmark** of optimization performance
- ❌ **Drop-in replacement** for official ProteinMPNN

**Use accordingly and validate independently.**

---

## Version History

- **2026-02-04**: Initial transparency report
  - Documented simulation methods
  - Clarified implementation status
  - Added recommendations for validation

---

**For questions or contributions to improve authenticity, please open an issue on GitHub.**
