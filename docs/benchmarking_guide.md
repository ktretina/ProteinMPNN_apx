# ProteinMPNN Benchmarking Guide

## Overview

This guide explains the benchmarking methodology used in ProteinMPNN_apx to evaluate model performance, accuracy, and efficiency.

## Benchmarking Metrics

### 1. Sequence Recovery Rate

**Definition**: The percentage of amino acids in the designed sequence that match the native sequence.

**Formula**:
```
Recovery = (Number of matching residues / Total residues) × 100%
```

**Interpretation**:
- **>40%**: Excellent recovery, highly accurate designs
- **30-40%**: Good recovery, acceptable for most applications
- **<30%**: Poor recovery, may indicate model issues

**Why it matters**: Higher recovery rates suggest the model understands protein structure-sequence relationships well.

### 2. Timing Metrics

**Measurements**:
- **Total inference time**: Time to generate all sequences
- **Per-sequence time**: Average time per designed sequence
- **Throughput**: Sequences generated per second

**Usage**: Compare timing across model variants to measure speedup:
```
Speedup = Baseline_time / Optimized_time
```

### 3. Burial Analysis

**Definition**: Separate recovery rates for buried vs. exposed residues.

**Categories**:
- **Buried residues**: <5% solvent accessible surface area (SASA)
- **Exposed residues**: >20% SASA
- **Intermediate**: 5-20% SASA

**Why it matters**:
- Buried residues are more constrained (hydrophobic core)
- Exposed residues have more freedom (surface interactions)
- Different optimization strategies may affect these differently

### 4. AlphaFold Validation (Optional)

**Process**:
1. Generate sequences with ProteinMPNN
2. Predict structures with AlphaFold2
3. Compare to native structure (TM-score, RMSD)

**Metrics**:
- **TM-score**: >0.8 indicates similar fold
- **RMSD**: <2Å indicates high structural similarity

**Why it matters**: Ultimate test of design quality - do the sequences fold correctly?

## Benchmarking Workflow

### Step 1: Prepare Data

```bash
# Place PDB files in data directory
cp path/to/*.pdb ./data/

# Or let the script download example structures
python benchmark.py  # Auto-downloads if data/ is empty
```

### Step 2: Run Baseline Benchmark

```bash
python benchmark.py \
  --model_variant baseline \
  --num_samples 10 \
  --output_dir ./output/baseline
```

### Step 3: Run Optimized Variant

```bash
python benchmark.py \
  --model_variant optimized_v1 \
  --num_samples 10 \
  --output_dir ./output/optimized_v1
```

### Step 4: Compare Results

Results are saved as JSON:

```json
{
  "total_time": 45.2,
  "avg_recovery": 38.5,
  "burial_analysis": {
    "buried": 42.1,
    "exposed": 35.8
  },
  "per_structure": [...]
}
```

## Best Practices

### 1. Consistent Test Set

- Use the same PDB files across all benchmarks
- Include diverse protein sizes and types
- Minimum 10 structures for statistical significance

### 2. Multiple Runs

Run each benchmark 3-5 times and report:
- Mean ± standard deviation
- Accounts for GPU variance and randomness

### 3. Controlled Environment

- Same hardware (GPU model, CUDA version)
- Same PyTorch version
- No other processes competing for GPU
- Warm-up run before timing (first run is often slower)

### 4. Report Full Context

When sharing results, include:
- Hardware specs (GPU model, memory)
- Software versions (PyTorch, CUDA, Python)
- Benchmark parameters (num_samples, batch_size)
- Date/time (driver updates can affect performance)

## Common Pitfalls

### 1. Cold Start Bias

**Problem**: First inference is slower due to kernel compilation.

**Solution**: Run a warm-up iteration before timing.

### 2. Batch Size Effects

**Problem**: Different batch sizes can significantly affect timing.

**Solution**: Use consistent batch size, report batch size used.

### 3. Memory Limitations

**Problem**: Large proteins or batch sizes may exceed GPU memory.

**Solution**:
- Use gradient checkpointing
- Reduce batch size
- Process large structures separately

### 4. Non-deterministic Operations

**Problem**: GPU operations may have slight randomness.

**Solution**:
- Use `torch.use_deterministic_algorithms(True)`
- Set seeds: `torch.manual_seed(42)`
- Report variance across multiple runs

## Advanced Benchmarking

### Profiling

Identify bottlenecks with PyTorch profiler:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model.predict(structure)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Usage

Track peak memory:

```python
torch.cuda.reset_peak_memory_stats()
model.predict(structure)
peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
```

### Scaling Analysis

Test how performance scales with:
- Protein size (residue count)
- Batch size
- Number of samples

## Interpreting Results

### Scenario 1: Faster but Lower Recovery

**Tradeoff**: Speed vs. accuracy
**Decision**: Acceptable if recovery stays >30% and speed gain is significant (>2x)

### Scenario 2: Slower but Higher Recovery

**Tradeoff**: Accuracy vs. speed
**Decision**: Only valuable if recovery improves substantially (>5%) and speed loss is minimal (<20%)

### Scenario 3: Faster AND Higher Recovery

**Outcome**: Clear win! This is the goal.

### Scenario 4: Slower AND Lower Recovery

**Outcome**: Optimization failed, debug or revert.

## AlphaFold Validation Workflow

### Setup

```bash
# Install AlphaFold (follow official instructions)
# Or use ColabFold for easier setup
pip install colabfold
```

### Running Validation

```bash
# Generate sequences
python benchmark.py --output_dir ./output/sequences

# Fold with AlphaFold
for seq in output/sequences/*.fasta; do
  colabfold_batch $seq ./output/alphafold/
done

# Compare structures (requires TMalign or similar)
for pdb in output/alphafold/*.pdb; do
  TMalign $pdb data/native.pdb
done
```

## Reporting Guidelines

### Minimum Report

Include in your README or paper:

```markdown
## Benchmark Results

**Hardware**: NVIDIA A100 40GB
**Software**: PyTorch 2.0.1, CUDA 11.8

| Variant | Recovery (%) | Time (s) | Speedup |
|---------|-------------|----------|---------|
| Baseline | 38.2 ± 1.1 | 45.3 ± 2.1 | 1.0x |
| Optimized | 37.8 ± 1.3 | 22.1 ± 1.5 | 2.05x |
```

### Full Report

Additionally include:
- Per-structure results
- Burial analysis breakdown
- Memory usage comparison
- Profiling highlights
- AlphaFold validation (if performed)

## Questions?

For issues or questions about benchmarking methodology, please open a GitHub issue.
