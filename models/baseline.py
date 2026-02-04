"""
Baseline ProteinMPNN Implementation

Reference implementation based on the original ProteinMPNN architecture.
Message-passing neural network for protein sequence design from backbone structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for residue positions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]


class MPNNLayer(nn.Module):
    """Message Passing Neural Network layer for encoding protein structure."""

    def __init__(self, node_features: int, edge_features: int, hidden_dim: int):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(node_features * 2 + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features)
        )

        self.norm = nn.LayerNorm(node_features)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_features]
            edge_index: [2, num_edges] - source and target node indices
            edge_features: [num_edges, edge_features]
        """
        src, dst = edge_index

        # Gather source and destination node features
        src_features = node_features[src]  # [num_edges, node_features]
        dst_features = node_features[dst]  # [num_edges, node_features]

        # Compute messages
        message_input = torch.cat([src_features, dst_features, edge_features], dim=-1)
        messages = self.message_net(message_input)  # [num_edges, hidden_dim]

        # Aggregate messages (sum over incoming edges)
        num_nodes = node_features.size(0)
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=node_features.device)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), messages)

        # Update node features
        update_input = torch.cat([node_features, aggregated], dim=-1)
        delta = self.update_net(update_input)

        # Residual connection and normalization
        return self.norm(node_features + delta)


class ProteinEncoder(nn.Module):
    """Encoder that processes protein backbone geometry into latent representations."""

    def __init__(
        self,
        node_features: int = 128,
        edge_features: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()

        self.node_embedding = nn.Linear(6, node_features)  # CA coords + orientation
        self.edge_embedding = nn.Linear(32, edge_features)  # RBF distance encoding

        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(node_features, edge_features, hidden_dim)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(node_features, hidden_dim)

    def forward(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_coords: [num_nodes, 6] - CA coordinates and orientation
            edge_index: [2, num_edges]
            edge_distances: [num_edges, 32] - RBF encoded distances

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        # Embed nodes and edges
        node_features = self.node_embedding(node_coords)
        edge_features = self.edge_embedding(edge_distances)

        # Apply message passing layers
        for mpnn_layer in self.mpnn_layers:
            node_features = mpnn_layer(node_features, edge_index, edge_features)

        # Project to output dimension
        return self.output_projection(node_features)


class AutoregressiveDecoder(nn.Module):
    """Autoregressive decoder for sequence generation."""

    def __init__(
        self,
        hidden_dim: int = 128,
        vocab_size: int = 20,  # 20 amino acids
        num_layers: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Token embedding for previously generated residues
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)  # +1 for mask token

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        encoder_output: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: [batch_size, seq_len, hidden_dim]
            tokens: [batch_size, seq_len] - previous tokens (None for generation)
            temperature: Sampling temperature

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, _ = encoder_output.shape

        if tokens is None:
            # Start with mask tokens
            tokens = torch.full((batch_size, seq_len), self.vocab_size,
                              dtype=torch.long, device=encoder_output.device)

        # Embed tokens
        token_emb = self.token_embedding(tokens)  # [batch_size, seq_len, hidden_dim]

        # Add positional encoding
        token_emb = token_emb.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        token_emb = self.pos_encoder(token_emb)
        token_emb = token_emb.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]

        # Create causal mask for autoregressive generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(encoder_output.device)

        # Decode
        memory = encoder_output
        decoder_output = self.transformer_decoder(
            token_emb,
            memory,
            tgt_mask=causal_mask
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def generate_sequence(
        self,
        encoder_output: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Autoregressively generate a sequence.

        Args:
            encoder_output: [batch_size, seq_len, hidden_dim]
            temperature: Sampling temperature

        Returns:
            sequence: [batch_size, seq_len] - amino acid indices
        """
        batch_size, seq_len, _ = encoder_output.shape
        device = encoder_output.device

        # Start with all mask tokens
        tokens = torch.full((batch_size, seq_len), self.vocab_size,
                          dtype=torch.long, device=device)

        # Generate one position at a time
        for pos in range(seq_len):
            # Get logits for current position
            logits = self.forward(encoder_output, tokens, temperature)

            # Sample from distribution at current position
            probs = F.softmax(logits[:, pos, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)

            # Update tokens
            tokens[:, pos] = next_token

        return tokens


class BaselineProteinMPNN(nn.Module):
    """
    Baseline ProteinMPNN model.

    Architecture:
    1. Encoder: Message-passing network over protein backbone graph
    2. Decoder: Autoregressive transformer for sequence generation
    """

    def __init__(
        self,
        node_features: int = 128,
        edge_features: int = 32,
        hidden_dim: int = 128,
        vocab_size: int = 20,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3
    ):
        super().__init__()

        self.encoder = ProteinEncoder(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers
        )

        self.decoder = AutoregressiveDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_decoder_layers
        )

    def forward(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Full forward pass: encode structure and generate sequence.

        Args:
            node_coords: [batch_size, num_nodes, 6]
            edge_index: [2, num_edges]
            edge_distances: [num_edges, 32]
            temperature: Sampling temperature

        Returns:
            sequences: [batch_size, num_nodes]
        """
        # Encode structure
        encoder_output = self.encoder(node_coords, edge_index, edge_distances)

        # Add batch dimension if needed
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(0)

        # Generate sequence
        sequences = self.decoder.generate_sequence(encoder_output, temperature)

        return sequences

    def encode_only(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor
    ) -> torch.Tensor:
        """Encode structure without sequence generation."""
        return self.encoder(node_coords, edge_index, edge_distances)


# Utility functions for graph construction

def rbf_encode_distances(distances: torch.Tensor, num_rbf: int = 32,
                        min_dist: float = 2.0, max_dist: float = 22.0) -> torch.Tensor:
    """
    Radial Basis Function encoding of distances.

    Args:
        distances: [num_edges] - pairwise distances
        num_rbf: Number of RBF kernels
        min_dist: Minimum distance
        max_dist: Maximum distance

    Returns:
        rbf_encoded: [num_edges, num_rbf]
    """
    mu = torch.linspace(min_dist, max_dist, num_rbf, device=distances.device)
    sigma = (max_dist - min_dist) / num_rbf

    distances = distances.unsqueeze(-1)  # [num_edges, 1]
    mu = mu.unsqueeze(0)  # [1, num_rbf]

    rbf = torch.exp(-((distances - mu) ** 2) / (2 * sigma ** 2))
    return rbf


def build_knn_graph(coords: torch.Tensor, k: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build k-nearest neighbor graph from CA coordinates.

    Args:
        coords: [num_nodes, 3] - CA coordinates
        k: Number of neighbors

    Returns:
        edge_index: [2, num_edges]
        distances: [num_edges]
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(coords, coords)  # [num_nodes, num_nodes]

    # Find k-nearest neighbors (excluding self)
    _, indices = torch.topk(dist_matrix, k + 1, largest=False, dim=-1)
    indices = indices[:, 1:]  # Remove self

    # Build edge index
    src = torch.arange(coords.size(0), device=coords.device).unsqueeze(-1).expand(-1, k)
    edge_index = torch.stack([src.flatten(), indices.flatten()], dim=0)

    # Get distances for edges
    distances = dist_matrix[edge_index[0], edge_index[1]]

    return edge_index, distances
