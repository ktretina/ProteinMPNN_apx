# ProteinMPNN Model Variants

This directory contains implementations of various ProteinMPNN model variants for benchmarking and optimization.

## Adding a New Model Variant

To add a new optimized version of ProteinMPNN:

1. Create a new Python file: `models/your_variant_name.py`
2. Implement the required interface (see below)
3. Update the benchmark script to include your variant
4. Document your optimizations

## Required Interface

Each model variant should implement the following interface:

```python
class YourVariantModel:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize the model.

        Args:
            checkpoint_path: Path to model weights
            device: Device to load model on ('cuda' or 'cpu')
        """
        pass

    def predict(self, structure, num_samples=5):
        """
        Generate sequences for a given structure.

        Args:
            structure: Protein structure (PDB or internal format)
            num_samples: Number of sequences to generate

        Returns:
            List of designed sequences
        """
        pass
```

## Optimization Ideas

Potential areas for optimization:

1. **Kernel Fusion**: Fuse multiple operations into single kernels
2. **Mixed Precision**: Use FP16/BF16 for faster inference
3. **Graph Optimization**: Use torch.compile or TorchScript
4. **Batch Processing**: Improve batching efficiency
5. **Memory Optimization**: Reduce memory footprint
6. **Custom CUDA Kernels**: Hand-optimized critical operations
7. **Model Pruning**: Remove unnecessary parameters
8. **Quantization**: INT8 or other quantization schemes

## Benchmarking Your Variant

After implementing a variant:

```bash
python benchmark.py --model_variant your_variant --output_dir ./output/your_variant
```

Compare results against the baseline implementation.

## Example Variants

### Baseline
- Original ProteinMPNN implementation
- Reference for comparison

### Optimized_v1 (Template)
- Add your first optimization here
- Document speedup and accuracy changes

### Optimized_v2 (Template)
- Add your second optimization here
- Document speedup and accuracy changes

## Documentation Requirements

For each variant, document:

1. **Optimization technique** used
2. **Expected speedup** (e.g., 2x faster)
3. **Accuracy impact** (if any)
4. **Memory usage** changes
5. **Hardware requirements** (e.g., specific GPU features)

## Testing

Ensure your variant:
- Produces valid protein sequences
- Maintains reasonable recovery rates (>30%)
- Handles edge cases (small proteins, large proteins)
- Works with both CPU and GPU

## Questions?

Open an issue or check the main project documentation.
