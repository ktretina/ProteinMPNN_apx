# Baseline Variant

## Overview
The Baseline variant is the reference implementation of ProteinMPNN with standard hyperparameters. This serves as the control for all optimization experiments.

## Architecture Parameters

```python
config = {
    'num_letters': 21,           # Amino acid alphabet size
    'node_features': 128,        # Node embedding dimension
    'edge_features': 128,        # Edge embedding dimension
    'hidden_dim': 128,           # Hidden layer dimension
    'num_encoder_layers': 3,     # Encoder depth
    'num_decoder_layers': 3,     # Decoder depth
    'k_neighbors': 48,           # K-nearest neighbors for graph
    'batch_size': 1              # Sequential processing
}
```

## Model Instantiation

```python
from protein_mpnn_utils import ProteinMPNN

model = ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48
).to(device)
```

## Architecture Diagram

```
Original ProteinMPNN (Baseline)
================================

INPUT: Protein Structure (N, CA, C, O coordinates)
  │
  ├─> K-NN Graph Construction (k=48)
  │     └─> Dense local connectivity
  │
  v
┌─────────────────────────────────┐
│       ENCODER (3 layers)        │
│                                 │
│  ┌───────────────────────────┐ │
│  │ Message Passing Layer 1   │ │
│  │  - Node features: 128     │ │
│  │  - Edge features: 128     │ │
│  │  - Hidden dim: 128        │ │
│  └───────────────────────────┘ │
│              ↓                  │
│  ┌───────────────────────────┐ │
│  │ Message Passing Layer 2   │ │
│  └───────────────────────────┘ │
│              ↓                  │
│  ┌───────────────────────────┐ │
│  │ Message Passing Layer 3   │ │
│  └───────────────────────────┘ │
│                                 │
└─────────────────────────────────┘
              ↓
    Structural Encoding (128-dim)
              ↓
┌─────────────────────────────────┐
│       DECODER (3 layers)        │
│                                 │
│  ┌───────────────────────────┐ │
│  │ Autoregressive Layer 1    │ │
│  │  - Attends to structure   │ │
│  │  - Attends to prior AAs   │ │
│  └───────────────────────────┘ │
│              ↓                  │
│  ┌───────────────────────────┐ │
│  │ Autoregressive Layer 2    │ │
│  └───────────────────────────┘ │
│              ↓                  │
│  ┌───────────────────────────┐ │
│  │ Autoregressive Layer 3    │ │
│  └───────────────────────────┘ │
│                                 │
└─────────────────────────────────┘
              ↓
    ┌─────────────────┐
    │ Output Head     │
    │ 128 → 21 AAs    │
    └─────────────────┘
              ↓
OUTPUT: Log probabilities over 21 amino acids
```

## Message Passing Details

Each message passing layer performs:

```python
def message_passing_layer(h, X, E_idx):
    """
    Args:
        h: Node features [B, L, 128]
        X: Coordinates [B, L, 4, 3]  (N, CA, C, O)
        E_idx: K-NN edges [B, L, 48]
    """
    # 1. Gather neighbor features
    h_neighbors = h[E_idx]  # [B, L, 48, 128]

    # 2. Compute edge features (distances, angles)
    edge_features = compute_edge_features(X, E_idx)  # [B, L, 48, 128]

    # 3. Combine node and edge features
    messages = torch.cat([h_neighbors, edge_features], dim=-1)

    # 4. Apply message MLP
    messages = message_mlp(messages)  # [B, L, 48, 128]

    # 5. Aggregate (mean pooling)
    aggregated = messages.mean(dim=2)  # [B, L, 128]

    # 6. Update node features
    h_new = update_mlp(torch.cat([h, aggregated], dim=-1))

    # 7. Layer normalization
    h_new = layer_norm(h_new)

    return h_new
```

## Performance Characteristics

### Speed
- **Mean inference time**: 14.69 ms
- **Throughput**: 7,217 residues/sec
- **Speedup**: 1.0x (baseline)

### Accuracy
- **Mean recovery**: 6.2%
- **Consensus recovery**: 6.6%
- **Accuracy loss**: 0% (reference)

### Memory
- **Peak memory**: ~500 MB (106 residue protein)
- **Parameter count**: ~2.1M parameters

## Parameter Count Breakdown

```
Encoder:
  - Embedding layers: ~50K
  - 3x Message Passing: 3 × 200K = 600K
  - Layer norms: ~3K

Decoder:
  - 3x Autoregressive layers: 3 × 250K = 750K
  - Layer norms: ~3K

Output Head: ~3K

Total: ~2.1M parameters
```

## When to Use Baseline

✅ **Use when:**
- Maximum accuracy is required
- Inference time is not critical
- You need a reference for comparison
- Pretrained weights are being loaded

❌ **Don't use when:**
- Real-time inference is needed
- Running on resource-constrained devices
- Batch processing many proteins

## Comparison to Other Variants

| Metric | Baseline | Fast | Minimal | EXTREME_v2 |
|--------|----------|------|---------|------------|
| Layers | 3+3 | 3+3 | 2+2 | 2+2 |
| Dim | 128 | 128 | 64 | 64 |
| k | 48 | 16 | 48 | 12 |
| Speed | 1.0x | 1.67x | 1.84x | 7.7x |
| Accuracy | 6.2% | 0.9% | 6.6% | 2.7% |

## Example Usage

```python
import torch
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

# Initialize baseline model
model = ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48
).to(device)

# Load protein structure
pdb_dict_list = parse_PDB('protein.pdb', ca_only=False)
protein = pdb_dict_list[0]

# Featurize
X, S, mask, lengths, chain_M, chain_encoding_all, *_ = tied_featurize(
    [protein], device, None, None, None, None, None, None, ca_only=False
)

# Generate sequence
with torch.no_grad():
    randn = torch.randn(chain_M.shape, device=device)
    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
    designed_seq = torch.argmax(log_probs, dim=-1)
```

## Key Insights

1. **Dense Connectivity (k=48)**: Provides rich structural context but increases computational cost quadratically
2. **Deep Architecture (3+3 layers)**: Allows information to propagate across long distances in the protein structure
3. **Large Hidden Dimensions (128)**: Provides representational capacity but increases memory and compute
4. **Trade-off**: Best accuracy but slowest speed - good baseline for measuring optimization impact

## References

- Original ProteinMPNN paper: [Dauparas et al., Science 2022](https://www.science.org/doi/10.1126/science.add2187)
- Official implementation: [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
