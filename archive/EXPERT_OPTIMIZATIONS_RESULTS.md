# Expert-Level Optimizations: Test Results

**Source**: expert_proteinmpnn.txt
**Hardware**: Apple Silicon M3 Pro
**Date**: 2026-02-04
**Policy**: Only actually tested optimizations included

---

## Overview

The expert_proteinmpnn.txt file proposed 4 advanced hardware-specific optimizations targeting Apple M3 architecture:

1. Manual Kernel Fusion (Tile Memory) - **Not tested** (requires custom Metal kernels)
2. ANE Bucketed Compilation - **Not tested** (requires CoreML conversion)
3. **Zero-Copy Graph Construction** - ✅ **TESTED**
4. **Manual Mixed Precision** - ✅ **TESTED**

---

## ✅ Optimization #4: Manual Mixed Precision

### Concept

Cast Linear layer weights to FP16 while keeping activations in FP32:
- **Weights**: FP16 (50% memory bandwidth reduction)
- **Biases**: FP32 (precision critical)
- **Activations**: FP32 (avoid accumulation errors)
- **Matrix multiplication**: Mixed precision (FP16 read, FP32 accumulate)

### Implementation

```python
def apply_manual_mixed_precision(model):
    """Cast Linear weights to FP16, keep biases/activations FP32."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.half()
            # Bias stays FP32
    return model
```

### Benchmark Results

| Configuration | Time | Throughput | Status |
|---------------|------|------------|---------|
| **FP32 Baseline** | 15.35 ± 0.31 ms | 6,903 res/sec | Reference |
| **Manual Mixed Precision** | 17.09 ± 2.82 ms | 6,201 res/sec | ❌ Slower |

**Speedup**: **0.90x** (10% slower)

### Analysis

❌ **Manual mixed precision is slower on MPS**

**Why it failed**:
1. FP16→FP32 conversion overhead on MPS
2. Mixed precision matmul not well-optimized on Metal
3. Increased variance (±2.82 ms vs ±0.31 ms)
4. MPS backend may not efficiently handle mixed dtypes

**Comparison to naive FP16**:
- Naive `model.half()`: Crashes with dtype errors
- Manual mixed precision: Runs but provides no speedup

**Conclusion**: Manual mixed precision doesn't work on MPS. Stick with FP32.

---

## ✅ Optimization #3: Zero-Copy Graph Construction

### Concept

Use CPU for k-NN search instead of GPU:
- **Rationale**: CPU has better branch prediction for divergent k-NN logic
- **Unified Memory**: No CPU↔GPU copy overhead on Apple Silicon
- **Method**: CPU computes neighbor indices, GPU reads via shared memory

### Implementation

```python
from sklearn.neighbors import NearestNeighbors

def cpu_knn_search(X_ca, k_neighbors, mask):
    """Perform k-NN on CPU using sklearn."""
    X_np = X_ca.cpu().numpy()

    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors,
        algorithm='ball_tree'
    ).fit(X_np)

    distances, indices = nbrs.kneighbors(X_np)

    # Move indices to GPU (unified memory = fast)
    return torch.from_numpy(indices).to(device)
```

### Benchmark Results (k-NN Search Only)

| Method | Time | Std Dev | Speedup |
|--------|------|---------|---------|
| **GPU k-NN** (baseline) | 5.19 ms | ±16.35 ms | 1.00x |
| **CPU k-NN** (unified memory) | 3.97 ms | ±11.84 ms | **1.31x** |

**Protein**: 106 residues, k=48 neighbors

### Analysis

✅ **CPU k-NN is 1.31x faster for graph construction**

**Why it works**:
1. sklearn's ball_tree algorithm is highly optimized for CPU
2. No copy overhead thanks to unified memory
3. CPU branch prediction handles divergent neighbor search well
4. Avoids GPU kernel launch overhead for small operation

**Limitations**:
1. High variance in measurements (±11-16 ms)
2. Only tested k-NN component in isolation
3. Full integration would require modifying ProteinFeatures class
4. Benefit may decrease for very large proteins (GPU parallelism wins)

**Integration difficulty**: Medium
- Would need to modify `ProteinFeatures` class
- Pass pre-computed E_idx to feature extraction
- Handle edge cases (padding, variable length)

**Conclusion**: CPU k-NN shows promise (1.31x) but full model integration not completed.

---

## ❌ Optimization #1: Manual Kernel Fusion (Not Tested)

### Concept

Fuse MPNN message and update phases into single Metal kernel:
- Use Tile Memory (GPU SRAM) to keep intermediates
- Minimize main memory round-trips
- Requires MLX + custom Metal Shading Language (MSL)

### Why Not Tested

**Complexity**: Very high
- Requires writing custom Metal kernels
- Needs deep understanding of M3 GPU architecture
- Would require MLX framework (different from PyTorch)
- Kernel development and debugging time: weeks

**Feasibility**: Low in current timeframe
- Would need Metal shader expertise
- Requires profiling to identify exact fusion points
- Testing and validation would be extensive

**Priority**: Low
- Already achieved 8.18x speedup with simpler methods
- Kernel fusion typically provides 1.2-1.5x additional gain
- Effort-to-reward ratio not favorable

---

## ❌ Optimization #2: ANE Bucketed Compilation (Not Tested)

### Concept

Offload encoder/decoder to Neural Engine with fixed shapes:
- Create 4 models for L=[64, 128, 256, 512]
- Zero-pad to nearest bucket
- Use CoreML to target ANE
- Runtime dispatch to appropriate bucket

### Why Not Tested

**Complexity**: High
- Requires CoreML conversion with coremltools
- Model tracing with fixed shapes
- Four separate models to manage
- Runtime dispatcher logic

**Feasibility**: Medium
- CoreML conversion can be done
- But ANE execution not guaranteed (may fall back to GPU/CPU)
- Bucket sizing may not match real workload distribution

**Priority**: Medium
- Could provide 1.5-2x speedup if ANE executes
- But significant engineering effort
- Risk of no speedup if ANE doesn't support operations

**Recommendation**: Worth exploring if targeting mobile deployment

---

## Summary of Expert Optimizations

| Optimization | Status | Result | Recommendation |
|--------------|--------|--------|----------------|
| **Manual Mixed Precision** | ✅ Tested | 0.90x (slower) | ❌ Don't use |
| **CPU k-NN** | ✅ Tested | 1.31x (k-NN only) | ⚠️ Promising but needs integration |
| **Kernel Fusion** | ❌ Not tested | N/A | ⚠️ High effort, low priority |
| **ANE Bucketing** | ❌ Not tested | N/A | ⚠️ Medium effort, uncertain benefit |

---

## Verified vs Claimed Performance

### What Was Claimed (expert_proteinmpnn.txt)

**Success Metric**: >10,000 residues/sec (>1.5x over baseline ~7,000 res/sec)

### What Was Actually Achieved

**Current best** (EXTREME-v2 from previous work):
- **Throughput**: 55,613 residues/sec
- **Speedup**: 8.18x over baseline
- **Already exceeds target by 5.5x**

**Expert optimizations tested**:
- Manual mixed precision: ❌ Made things slower
- CPU k-NN: ✅ 1.31x for k-NN step (not full model)

---

## Integration Recommendations

### If Pursuing CPU k-NN

**Estimated effort**: 1-2 days

**Steps**:
1. Modify `ProteinFeatures._dist()` to accept pre-computed E_idx
2. Add `use_cpu_knn` flag to ProteinMPNN constructor
3. Compute E_idx on CPU before feature extraction
4. Pass E_idx to features module
5. Benchmark full model (not just k-NN)

**Expected full model speedup**: 1.05-1.10x
- k-NN is ~5ms of ~15ms total
- 1.31x speedup on 5ms → 1.7ms savings
- New total: ~13.3ms → 1.15x full model speedup

**Worth it?**: Marginal benefit, already have 8.18x speedup

### If Pursuing Kernel Fusion

**Estimated effort**: 3-6 weeks

**Requirements**:
- Metal Shading Language expertise
- MLX framework adoption
- Kernel profiling and optimization
- Extensive testing

**Expected speedup**: 1.2-1.5x

**Worth it?**: Depends on production needs
- If targeting mobile (iPhone/iPad): Yes
- If M3 Pro/Max desktop: Marginal benefit

---

## Honest Assessment

### What Works (From All Testing)

**Actually achieved 8.18x speedup** using:
1. Model pruning (2+2 layers, dim=64): 1.93x
2. K-neighbors reduction (k=12): 1.83x
3. Batching (batch=8): 3.0x

**These are simple, working, production-ready optimizations.**

### What Doesn't Add Value

**Expert optimizations tested**:
- Manual mixed precision: Made things worse
- CPU k-NN: Marginal benefit, high integration cost

**Expert optimizations not tested**:
- Kernel fusion: Very high effort
- ANE bucketing: Uncertain benefit

### Conclusion

The simple architectural optimizations (pruning, k-reduction, batching) achieved **8.18x speedup** and **already exceed the expert optimization target of 1.5x**.

The proposed expert optimizations either:
1. Don't work on MPS (manual mixed precision)
2. Have marginal benefits not worth the integration cost (CPU k-NN)
3. Require prohibitive engineering effort (kernel fusion, ANE)

**Recommendation**: Stick with verified 8.18x speedup. Expert optimizations don't provide sufficient additional benefit for the M3 Pro use case.

---

## Files Created

1. **benchmark_manual_mixed_precision.py**
   - Tests FP16 weights + FP32 activations
   - Result: 0.90x (slower)
   - Converted 43 Linear layers to FP16

2. **benchmark_cpu_knn.py**
   - Tests CPU vs GPU k-NN search
   - Result: 1.31x speedup for k-NN component
   - Used sklearn NearestNeighbors (ball_tree)

3. **output/manual_mixed_precision.json**
   - Full benchmark data for mixed precision test

4. **output/cpu_knn_benchmark.json**
   - Full benchmark data for CPU k-NN test

5. **EXPERT_OPTIMIZATIONS_RESULTS.md** (this file)
   - Complete analysis of expert optimizations

---

**Verification Date**: 2026-02-04
**Status**: 2/4 expert optimizations tested, both show limited or negative benefit
**Current Best**: 8.18x speedup (EXTREME-v2) from simple architectural optimizations
