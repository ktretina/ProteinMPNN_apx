# ProteinMPNN_apx

A comprehensive benchmarking and optimization suite for ProteinMPNN model variants.

## Overview

ProteinMPNN_apx is a project focused on optimizing and benchmarking various implementations of the ProteinMPNN AI model for protein sequence design. This repository provides tools to measure performance, accuracy, and efficiency across different model variants.

## Features

- **Comprehensive Benchmarking**: Measure timing, recovery rates, and burial analysis
- **Model Variants**: Framework for testing different ProteinMPNN optimizations
- **Automated Evaluation**: Built-in metrics and AlphaFold validation workflow
- **Reproducible Results**: Standardized benchmarking suite with consistent metrics

## Project Structure

```
ProteinMPNN_apx/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── benchmark.py                 # Main benchmarking script
├── models/                      # Directory for model implementations
│   ├── __init__.py
│   └── README.md               # Model variants documentation
├── data/                        # Data directory
│   └── .gitkeep
├── output/                      # Benchmark results
│   └── .gitkeep
├── notebooks/                   # Jupyter notebooks for analysis
│   └── .gitkeep
└── docs/                        # Additional documentation
    └── benchmarking_guide.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ktretina/ProteinMPNN_apx.git
cd ProteinMPNN_apx
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Benchmarks

Basic benchmark run:
```bash
python benchmark.py
```

Benchmark with custom PDB files:
```bash
python benchmark.py --data_dir ./data --output_dir ./output
```

Advanced options:
```bash
python benchmark.py --num_samples 10 --batch_size 8 --device cuda
```

### Command Line Arguments

- `--data_dir`: Directory containing PDB files (default: ./data)
- `--output_dir`: Directory for benchmark results (default: ./output)
- `--num_samples`: Number of sequences to generate per structure (default: 5)
- `--batch_size`: Batch size for inference (default: 4)
- `--device`: Device to use (cuda/cpu, default: auto-detect)
- `--model_path`: Path to custom model weights

### Metrics

The benchmarking suite evaluates:

1. **Recovery Rates**: Sequence recovery compared to native sequences
2. **Timing**: Inference speed and throughput
3. **Burial Analysis**: Recovery rates for buried vs. exposed residues
4. **AlphaFold Validation**: Structural similarity of designed sequences (optional)

## Adding Model Variants

To add a new optimized ProteinMPNN variant:

1. Implement your model in `models/your_variant.py`
2. Follow the interface defined in `models/README.md`
3. Update benchmark.py to include your variant
4. Run benchmarks and compare results

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/optimization`)
3. Commit your changes (`git commit -m 'Add new optimization'`)
4. Push to the branch (`git push origin feature/optimization`)
5. Open a Pull Request

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, falls back to CPU)
- Git and GitHub CLI for repository management

## Documentation

For detailed information, see:

- [Benchmarking Guide](docs/benchmarking_guide.md) - Detailed benchmarking methodology
- [Model Variants](models/README.md) - Guide for implementing new variants

## License

This project is open source. Please add appropriate license information.

## Acknowledgments

Based on the original ProteinMPNN implementation. Model weights are downloaded from the official ProteinMPNN repository.

## Citation

If you use this benchmarking suite, please cite the original ProteinMPNN paper and this repository.
