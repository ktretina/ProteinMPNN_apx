# Implementation Guide: Placeholder vs. Real Code

**Purpose**: This document explains the difference between the **architectural demonstrations** in this repository and what a **production-ready implementation** requires.

---

## Quick Reference Table

| Component | Current Implementation | Real Implementation Required |
|-----------|----------------------|----------------------------|
| **Feature Extraction** | `torch.randn(N, 128)` | RBF encoding, orientations, residue types |
| **Graph Construction** | Assumed `edge_index` | Build k-NN from coordinates |
| **Timing** | Simulated from factors | `time.perf_counter()` with GPU sync |
| **Memory** | Calculated from tensors | Runtime profiling with `tracemalloc` |
| **Accuracy** | Assumed from literature | Validation on benchmark datasets |

---

## Section 1: Feature Extraction

### Current (Placeholder) Code

Found in most model files:

```python
# models/ultimate_pytorch.py, line ~230
def forward(self, coords, edge_index, distances):
    # Create features (placeholder)
    features = torch.randn(B, N, 128, device=coords.device, dtype=coords.dtype)
    x = self.input_proj(features)
```

### What's Wrong:
- ❌ Uses random numbers instead of actual geometric features
- ❌ Ignores coordinate information completely
- ❌ No residue type information
- ❌ No distance-based features

### Real Implementation Required

See `models/reference_implementation.py` for complete code:

```python
def extract_protein_features(coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    """REAL feature extraction from coordinates."""

    # 1. Build k-NN graph from coordinates
    edge_index, distances = build_knn_graph(coords, k=30)

    # 2. RBF encode distances
    rbf_distances = rbf_encode_distances(distances, d_min=0.0, d_max=20.0, d_count=16)

    # 3. Compute orientations
    orientations = compute_orientations(coords, edge_index)

    # 4. Positional encoding
    positions = torch.arange(N)
    pos_encoding = sin_cos_encoding(positions)

    # 5. Geometric features (center of mass distance, etc.)
    dist_to_com = compute_com_distances(coords)

    return {
        'node_features': torch.cat([pos_encoding, dist_to_com], dim=-1),
        'edge_features': torch.cat([rbf_distances, orientations], dim=-1),
        'edge_index': edge_index
    }
```

**Key Functions Needed**:
1. `rbf_encode_distances()` - Radial basis function encoding
2. `build_knn_graph()` - k-NN graph from coordinates
3. `compute_orientations()` - Edge direction vectors
4. `sin_cos_encoding()` - Positional embeddings

---

## Section 2: Graph Construction

### Current (Placeholder) Code

```python
# Assumes edge_index is provided
def forward(self, coords, edge_index, distances):
    # edge_index is passed in, not computed
    pass
```

### What's Wrong:
- ❌ Graph structure is assumed to exist
- ❌ No actual distance computation from coordinates
- ❌ Missing O(N²) → O(Nk) optimization

### Real Implementation Required

```python
def build_knn_graph(coords: torch.Tensor, k: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build k-NN graph from 3D coordinates.

    This is the O(N²) operation mentioned in the reference document.
    """
    N = coords.shape[0]

    # Compute pairwise distances (O(N²))
    dist_matrix = torch.cdist(coords, coords)  # [N, N]

    # Mask self-connections
    dist_matrix = dist_matrix + torch.eye(N, device=coords.device) * 1e6

    # Get k nearest neighbors
    nearest_dists, nearest_indices = torch.topk(
        dist_matrix, k, dim=1, largest=False
    )  # [N, k]

    # Build edge list
    src = torch.arange(N).unsqueeze(1).expand(-1, k)  # [N, k]
    dst = nearest_indices  # [N, k]

    edge_index = torch.stack([src.flatten(), dst.flatten()])  # [2, N*k]
    distances = nearest_dists.flatten()  # [N*k]

    return edge_index, distances
```

**Optimization Strategies**:
- For GPU: Vectorized `torch.cdist` (implemented above)
- For very long sequences: Spatial hashing to reduce O(N²)
- For CPU: KD-tree from `scipy.spatial`

---

## Section 3: Benchmarking

### Current (Simulated) Code

All benchmark JSON files use:

```python
# Simulation method (NOT actual measurement)
baseline_time = seq_length / 40.8  # Assumed baseline throughput
speedup_factor = 22.47  # From literature
optimized_time = baseline_time / speedup_factor
throughput = seq_length / optimized_time
```

### What's Wrong:
- ❌ No actual timing measurements
- ❌ No GPU synchronization
- ❌ No statistical confidence intervals
- ❌ No warmup runs

### Real Implementation Required

See `models/reference_implementation.py`:

```python
class RealBenchmark:
    def benchmark_inference(self, coords, num_runs=100, warmup_runs=10):
        """REAL timing with proper GPU synchronization."""

        # Warmup (CRITICAL for GPU benchmarking)
        for _ in range(warmup_runs):
            _ = self.model(coords)
            torch.mps.synchronize()  # Wait for GPU completion

        # Actual timing
        times = []
        for _ in range(num_runs):
            torch.mps.synchronize()  # Ensure GPU ready
            start = time.perf_counter()

            _ = self.model(coords)

            torch.mps.synchronize()  # Wait for GPU completion
            end = time.perf_counter()

            times.append(end - start)

        # Statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = seq_length / mean_time

        return {'mean_time': mean_time, 'std_time': std_time, 'throughput': throughput}
```

**Critical Steps**:
1. **Warmup runs**: First runs include compilation overhead
2. **GPU synchronization**: `torch.mps.synchronize()` or `torch.cuda.synchronize()`
3. **Multiple runs**: Get statistical confidence (100+ runs recommended)
4. **Proper timing**: `time.perf_counter()` for high resolution

---

## Section 4: Memory Profiling

### Current (Calculated) Code

```python
# Estimation method (NOT measured)
memory_mb = seq_length * hidden_dim * 4 / 1e6  # FP32 = 4 bytes
```

### What's Wrong:
- ❌ Only counts explicit tensors
- ❌ Ignores framework overhead
- ❌ Ignores fragmentation
- ❌ Ignores gradient buffers, optimizer state

### Real Implementation Required

```python
import tracemalloc

def profile_memory(model, coords):
    """REAL memory profiling."""

    # Start tracing
    tracemalloc.start()

    # Forward pass
    output = model(coords)

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # For GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1e6
    elif torch.backends.mps.is_available():
        # MPS doesn't have memory_allocated API
        # Would need Metal profiling tools
        gpu_memory = None

    return {
        'peak_cpu_mb': peak / 1e6,
        'current_cpu_mb': current / 1e6,
        'peak_gpu_mb': gpu_memory
    }
```

**Tools Needed**:
- Python: `tracemalloc`
- PyTorch CUDA: `torch.cuda.memory_allocated()`
- PyTorch MPS: Metal profiling tools (Instruments.app)
- System: `nvidia-smi` (NVIDIA) or Activity Monitor (Mac)

---

## Section 5: Accuracy Validation

### Current (Assumed) Claims

From documentation:
- "BFloat16: <0.5% accuracy loss"
- "Flash Attention: 0% loss (mathematically equivalent)"
- "Maintains >99% baseline accuracy"

### What's Wrong:
- ❌ No actual validation performed
- ❌ Claims based on literature, not verified
- ❌ Assumes transfer to ProteinMPNN architecture

### Real Implementation Required

```python
def validate_accuracy(optimized_model, baseline_model, test_proteins):
    """REAL accuracy validation."""

    results = []

    for protein in test_proteins:
        # Baseline prediction
        with torch.no_grad():
            baseline_logits = baseline_model(protein.coords)
            baseline_seq = torch.argmax(baseline_logits, dim=-1)

        # Optimized prediction
        with torch.no_grad():
            optimized_logits = optimized_model(protein.coords)
            optimized_seq = torch.argmax(optimized_logits, dim=-1)

        # Compare
        sequence_identity = (baseline_seq == optimized_seq).float().mean().item()

        # Perplexity comparison
        baseline_ppl = compute_perplexity(baseline_logits, protein.true_sequence)
        optimized_ppl = compute_perplexity(optimized_logits, protein.true_sequence)

        results.append({
            'sequence_identity': sequence_identity,
            'baseline_perplexity': baseline_ppl,
            'optimized_perplexity': optimized_ppl,
            'perplexity_ratio': optimized_ppl / baseline_ppl
        })

    # Aggregate statistics
    mean_identity = np.mean([r['sequence_identity'] for r in results])
    mean_ppl_ratio = np.mean([r['perplexity_ratio'] for r in results])

    return {
        'mean_sequence_identity': mean_identity,
        'mean_perplexity_ratio': mean_ppl_ratio,
        'accuracy_loss_percent': (1 - mean_identity) * 100
    }
```

**Requirements**:
1. Official ProteinMPNN checkpoint
2. Benchmark dataset (CASP14, CATH, etc.)
3. Ground truth sequences
4. Statistical significance testing

---

## Section 6: Complete Example

### Placeholder Implementation (Current)

```python
# models/ultimate_pytorch.py
model = UltimatePyTorchProteinMPNN(hidden_dim=128)
coords = torch.randn(1, 100, 3)  # Random coordinates
logits = model(coords, None, None)  # Missing edge_index, distances
# Benchmark: 22.47x speedup (simulated)
```

### Real Implementation (Required)

```python
# Complete pipeline
from models.reference_implementation import RealProteinMPNN, RealBenchmark

# 1. Load real protein coordinates
coords = load_pdb_coordinates("1ubq.pdb")  # [N, 3] in Angstroms

# 2. Create model
model = RealProteinMPNN(hidden_dim=128)
model = model.to('mps')

# 3. Forward pass (complete feature extraction)
logits = model(coords)  # Internally: extract_protein_features → encode → decode

# 4. Benchmark (real timing)
benchmark = RealBenchmark(model, torch.device('mps'))
results = benchmark.benchmark_inference(coords, num_runs=100)

print(f"Mean time: {results['mean_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
print(f"Throughput: {results['throughput_res_per_sec']:.1f} res/sec")

# 5. Memory profiling
memory = profile_memory(model, coords)
print(f"Peak memory: {memory['peak_cpu_mb']:.1f} MB")

# 6. Accuracy validation
accuracy = validate_accuracy(model, baseline_model, test_set)
print(f"Accuracy loss: {accuracy['accuracy_loss_percent']:.2f}%")
```

---

## Section 7: What You Can Trust

### ✅ Trustworthy (Architectural Patterns)

1. **Optimization Strategies**: The techniques (Flash Attention, KV Caching, etc.) are real and valid
2. **PyTorch/MLX Structure**: Module hierarchy and device selection logic is correct
3. **Theoretical Analysis**: Complexity analysis (O(N²) → O(N)) is sound
4. **Literature References**: Speedup factors are from published research

### ❌ Not Trustworthy (Needs Validation)

1. **Specific Speedup Numbers**: "22.47x" needs actual measurement on your hardware
2. **Throughput Metrics**: "925.9 res/sec" is calculated, not measured
3. **Memory Usage**: "118 MB" is estimated, not profiled
4. **Accuracy Claims**: "<1% loss" needs validation on ProteinMPNN specifically

---

## Section 8: How to Convert to Real Implementation

### Step-by-Step Process:

1. **Start with Reference Implementation**:
   ```bash
   python models/reference_implementation.py
   ```
   This shows REAL feature extraction and benchmarking

2. **Integrate Real Features**:
   Replace placeholder code in optimization variants with real extraction:
   ```python
   # Replace this:
   features = torch.randn(B, N, 128)

   # With this:
   features = extract_protein_features(coords)
   ```

3. **Add Proper Benchmarking**:
   Use `RealBenchmark` class for all timing measurements

4. **Validate on Test Set**:
   - Download official ProteinMPNN checkpoint
   - Run on CASP/CATH benchmark
   - Compare sequence recovery rates

5. **Document Real Results**:
   Replace simulated JSON with actual measurements

---

## Conclusion

This repository provides:
- ✅ **Educational value**: Demonstrates optimization architecture
- ✅ **Code patterns**: Shows how to structure optimizations
- ✅ **Theoretical foundation**: Explains why optimizations work

This repository does NOT provide:
- ❌ **Production code**: Feature extraction is placeholder
- ❌ **Validated benchmarks**: Numbers are simulated
- ❌ **Drop-in replacement**: Requires significant development

**Use as a starting point and reference, not as a complete solution.**

---

For questions or contributions to add real implementations:
- Read: `TRANSPARENCY_REPORT.md`
- Example: `models/reference_implementation.py`
- Contribute: Open a pull request with validated code
