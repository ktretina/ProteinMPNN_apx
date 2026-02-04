# Archive Folder

**This folder contains files from the development process that are not part of the final analysis.**

Date archived: 2026-02-04

---

## What's Here

### Deprecated Documentation (17 .md files)

Historical documentation that has been superseded by the final 6 documents in the parent directory.

**Why archived**:
- Redundant content
- Outdated progress reports
- Superseded by consolidated documentation
- Multiple overlapping "final" summaries

**What superseded them**:
- `README.md` - Main entry point with all key info
- `FINAL_RESULTS_WITH_ACCURACY.md` - Complete technical story
- `EXECUTIVE_SUMMARY.md` - High-level narrative
- `EXPERT_OPTIMIZATIONS_FINAL.md` - Expert optimization details
- `DOCUMENTATION_INDEX.md` - Navigation guide
- `DOCUMENTATION_CLEANUP_SUMMARY.md` - Cleanup explanation

### Unused Python Scripts (17 .py files)

Benchmark and implementation scripts that were superseded by more comprehensive testing.

#### Early Benchmark Scripts (7 files)
- `benchmark.py` - Original benchmark
- `benchmark_variants.py` - Early variant testing
- `benchmark_variants_simple.py` - Simplified early testing
- `benchmark_optimized_variants.py` - Early optimization tests
- `benchmark_comprehensive.py` - Early comprehensive (superseded)
- `official_proteinmpnn_benchmark.py` - Baseline only
- `run_real_benchmarks.py` - Early runner

**Why archived**: Superseded by `benchmark_accuracy_comprehensive.py` which tests all variants with both speed AND accuracy.

#### Component Benchmark Scripts (6 files)
- `benchmark_batching.py` - Batching only
- `benchmark_model_pruning.py` - Pruning only
- `benchmark_cpu_knn.py` - CPU k-NN component only
- `benchmark_extreme_k_reduction.py` - k-value sweep
- `benchmark_ultimate_variants.py` - Ultimate/EXTREME variants
- `benchmark_extreme_v2.py` - EXTREME-v2 (kept in parent for reference)

**Why archived**: Individual component tests superseded by comprehensive testing. EXTREME-v2 kept as it shows the 8.18x result clearly.

#### Failed Optimization Scripts (4 files)
- `benchmark_compile.py` - torch.compile (0.99x - failed)
- `benchmark_quantization.py` - Int8 quantization (not supported on MPS)
- `benchmark_variants_fp16.py` - FP16 (errors)
- `benchmark_manual_mixed_precision.py` - Manual mixed (0.90x - slower)

**Why archived**: These optimizations failed. Results documented in final docs, but scripts archived since they're not useful.

#### Training Script (1 file)
- `train_distillation.py` - Knowledge distillation training

**Why archived**: Training failed (NaN losses, data loading issues). Framework implementation exists but no verified speedup.

---

## What's NOT Archived (Active Files)

### Documentation (6 .md files)
- `README.md` - Main entry point
- `FINAL_RESULTS_WITH_ACCURACY.md` - Complete technical details
- `EXECUTIVE_SUMMARY.md` - Big picture story
- `EXPERT_OPTIMIZATIONS_FINAL.md` - Expert optimization details
- `DOCUMENTATION_INDEX.md` - Navigation guide
- `DOCUMENTATION_CLEANUP_SUMMARY.md` - This cleanup explained

### Python Scripts (9 .py files)

**Core Benchmarking**:
- `benchmark_accuracy_comprehensive.py` - **Primary benchmark** (all variants with accuracy)
- `benchmark_extreme_v2.py` - EXTREME-v2 demonstration (8.18x)

**Expert Implementations**:
- `implement_ane_bucketing.py` - ANE implementation (2.75x avg)
- `implement_kernel_fusion_mlx.py` - Kernel fusion MLX (1.28x)
- `implement_cpu_knn_full.py` - CPU k-NN integration

**Research & Analysis**:
- `ane_bucketing_research.py` - ANE research
- `kernel_fusion_research.py` - Kernel fusion research
- `kernel_fusion_analysis.py` - Kernel fusion analysis
- `implement_kernel_fusion.py` - Kernel fusion design

---

## File Counts

**Archived**:
- 17 deprecated .md files
- 17 unused .py files
- **Total: 34 files**

**Active** (in parent directory):
- 6 active .md files
- 9 active .py files
- **Total: 15 files**

**Reduction**: 34 archived + 15 active = 49 total → 69% archived

---

## Should You Use These?

**No** - These files are kept for historical reference only.

**Instead**:
- For benchmarking: Use `benchmark_accuracy_comprehensive.py`
- For documentation: Read `README.md` first
- For expert optimizations: See `EXPERT_OPTIMIZATIONS_FINAL.md`
- For results: Check `output/*.json` files

---

## Why Keep Them?

1. **Historical record**: Shows the development process
2. **Failed optimizations**: Code for optimizations that didn't work (useful to know what NOT to try)
3. **Component tests**: May be useful for isolating specific components in future
4. **Reproducibility**: Complete record of all work done

---

## Migration Guide

If you have scripts or documentation referencing archived files:

### Old Script → New Script
- `benchmark_comprehensive.py` → `benchmark_accuracy_comprehensive.py`
- `benchmark_batching.py` → `benchmark_accuracy_comprehensive.py`
- `benchmark_model_pruning.py` → `benchmark_accuracy_comprehensive.py`
- `benchmark_cpu_knn.py` → `implement_cpu_knn_full.py`
- `benchmark_ultimate_variants.py` → `benchmark_extreme_v2.py`

### Old Doc → New Doc
- `VERIFIED_RESULTS_SUMMARY.md` → `FINAL_RESULTS_WITH_ACCURACY.md`
- `COMPLETE_FINAL_SUMMARY.md` → `EXECUTIVE_SUMMARY.md`
- `EXPERT_OPTIMIZATIONS_RESULTS.md` → `EXPERT_OPTIMIZATIONS_FINAL.md`
- `WHAT_TO_DO_NEXT.md` → `README.md`
- All others → See `DOCUMENTATION_INDEX.md`

---

## Summary

This archive contains the **development history** but not the **final results**.

For actual use:
- ✅ Use files in parent directory
- ❌ Don't use files in this archive

---

**Archived**: 2026-02-04
**Reason**: Documentation cleanup and consolidation
**Status**: Historical reference only
