"""
KV Caching Optimized ProteinMPNN

Optimization: Implements Key-Value caching for attention mechanism.

Key benefits:
- Reduces attention complexity from O(L²) to O(L) per step
- Avoids recomputing K,V for previous tokens
- Pre-allocated buffers prevent memory fragmentation
- Essential for long protein sequences

Reference: Section 7 of optimization document
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from models.baseline import ProteinEncoder, PositionalEncoding
import math


class KVCache:
    """
    Key-Value cache for transformer attention.

    Pre-allocates fixed-size buffers to avoid dynamic memory allocation
    during autoregressive generation.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize KV cache buffers.

        Args:
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            device: Device to allocate on
            dtype: Data type
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Pre-allocate cache buffers
        # Shape: [batch_size, num_heads, max_seq_len, head_dim]
        self.key_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.value_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )

        # Track current position
        self.current_pos = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key and value at position.

        Args:
            key: [batch_size, num_heads, 1, head_dim] - new key
            value: [batch_size, num_heads, 1, head_dim] - new value
            pos: Position to update

        Returns:
            keys: [batch_size, num_heads, pos+1, head_dim] - all keys up to pos
            values: [batch_size, num_heads, pos+1, head_dim] - all values up to pos
        """
        batch_size = key.size(0)

        # Write to cache at position
        self.key_cache[:batch_size, :, pos:pos + 1, :] = key
        self.value_cache[:batch_size, :, pos:pos + 1, :] = value

        # Return all cached keys/values up to current position
        keys = self.key_cache[:batch_size, :, :pos + 1, :]
        values = self.value_cache[:batch_size, :, :pos + 1, :]

        return keys, values

    def reset(self):
        """Reset cache to initial state."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.current_pos = 0


class CachedMultiHeadAttention(nn.Module):
    """Multi-head attention with KV caching support."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position: Optional[int] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attention with optional KV caching.

        Args:
            query: [batch_size, query_len, hidden_dim]
            key: [batch_size, key_len, hidden_dim]
            value: [batch_size, value_len, hidden_dim]
            kv_cache: Optional KV cache
            position: Current position (for cache)
            mask: Attention mask

        Returns:
            output: [batch_size, query_len, hidden_dim]
        """
        batch_size, query_len, _ = query.shape

        # Project Q, K, V
        Q = self.q_proj(query)  # [batch_size, query_len, hidden_dim]
        K = self.k_proj(key)    # [batch_size, key_len, hidden_dim]
        V = self.v_proj(value)  # [batch_size, value_len, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Use cache if provided
        if kv_cache is not None and position is not None:
            # For autoregressive generation, query_len should be 1
            assert query_len == 1, "KV cache only supports single-position queries"

            # Update cache and get all cached K, V
            K, V = kv_cache.update(K, V, position)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attn_weights, V)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        output = self.out_proj(output)

        return output


class CachedDecoderLayer(nn.Module):
    """Transformer decoder layer with KV caching."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int):
        super().__init__()

        self.self_attn = CachedMultiHeadAttention(hidden_dim, num_heads)
        self.cross_attn = CachedMultiHeadAttention(hidden_dim, num_heads)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_kv_cache: Optional[KVCache] = None,
        cross_kv_cache: Optional[KVCache] = None,
        position: Optional[int] = None,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder layer forward with optional caching.

        Args:
            x: [batch_size, seq_len, hidden_dim]
            encoder_output: [batch_size, enc_len, hidden_dim]
            self_kv_cache: Cache for self-attention
            cross_kv_cache: Cache for cross-attention
            position: Current position
            causal_mask: Causal mask for self-attention

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        # Self-attention with residual
        attn_out = self.self_attn(
            x, x, x,
            kv_cache=self_kv_cache,
            position=position,
            mask=causal_mask
        )
        x = self.norm1(x + attn_out)

        # Cross-attention with encoder output
        cross_out = self.cross_attn(
            x, encoder_output, encoder_output,
            kv_cache=cross_kv_cache,
            position=position
        )
        x = self.norm2(x + cross_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        return x


class KVCachedDecoder(nn.Module):
    """Autoregressive decoder with KV caching for efficient generation."""

    def __init__(
        self,
        hidden_dim: int = 128,
        vocab_size: int = 20,
        num_layers: int = 3,
        num_heads: int = 8,
        max_seq_len: int = 2000
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)

        # Decoder layers
        ff_dim = hidden_dim * 4
        self.decoder_layers = nn.ModuleList([
            CachedDecoderLayer(hidden_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Pre-allocated KV caches (initialized on first use)
        self.caches = None

    def _init_caches(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize KV caches for all layers."""
        head_dim = self.hidden_dim // self.num_heads

        self.caches = []
        for _ in range(self.num_layers):
            # Self-attention cache
            self_cache = KVCache(
                batch_size, self.max_seq_len, self.num_heads, head_dim,
                device, dtype
            )
            # Cross-attention cache (encoder output doesn't change)
            cross_cache = KVCache(
                batch_size, self.max_seq_len, self.num_heads, head_dim,
                device, dtype
            )
            self.caches.append((self_cache, cross_cache))

    def generate_sequence(
        self,
        encoder_output: torch.Tensor,
        temperature: float = 0.1,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate sequence with optional KV caching.

        Args:
            encoder_output: [batch_size, seq_len, hidden_dim]
            temperature: Sampling temperature
            use_cache: Whether to use KV caching

        Returns:
            sequences: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = encoder_output.shape
        device = encoder_output.device
        dtype = encoder_output.dtype

        # Initialize caches if using cache
        if use_cache:
            self._init_caches(batch_size, device, dtype)

        # Start with all mask tokens
        tokens = torch.full((batch_size, seq_len), self.vocab_size,
                          dtype=torch.long, device=device)

        # Generate one position at a time
        for pos in range(seq_len):
            if use_cache:
                # Only process current position with cache
                current_token = tokens[:, pos:pos + 1]
                token_emb = self.token_embedding(current_token)

                # Apply positional encoding
                token_emb = token_emb.permute(1, 0, 2)
                token_emb = self.pos_encoder(token_emb)
                token_emb = token_emb.permute(1, 0, 2)

                # Pass through decoder layers with cache
                x = token_emb
                for layer_idx, layer in enumerate(self.decoder_layers):
                    self_cache, cross_cache = self.caches[layer_idx]
                    x = layer(x, encoder_output, self_cache, cross_cache, pos)

                # Get logits for current position
                logits = self.output_projection(x[:, 0, :])
            else:
                # Process all positions up to current (no cache)
                current_tokens = tokens[:, :pos + 1]
                token_emb = self.token_embedding(current_tokens)

                # Positional encoding
                token_emb = token_emb.permute(1, 0, 2)
                token_emb = self.pos_encoder(token_emb)
                token_emb = token_emb.permute(1, 0, 2)

                # Decoder layers
                x = token_emb
                for layer in self.decoder_layers:
                    x = layer(x, encoder_output)

                # Get logits for last position
                logits = self.output_projection(x[:, -1, :])

            # Sample from distribution
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)

            # Update tokens
            tokens[:, pos] = next_token

        return tokens


class KVCachedProteinMPNN(nn.Module):
    """
    ProteinMPNN with KV caching optimization.

    Complexity reduction:
    - Without cache: O(L²) per position → O(L³) total
    - With cache: O(L) per position → O(L²) total

    For L=500, this is a ~500x speedup in theory.
    """

    def __init__(
        self,
        node_features: int = 128,
        edge_features: int = 32,
        hidden_dim: int = 128,
        vocab_size: int = 20,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        max_seq_len: int = 2000
    ):
        super().__init__()

        self.encoder = ProteinEncoder(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers
        )

        self.decoder = KVCachedDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

    def forward(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        temperature: float = 0.1,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional KV caching.

        Args:
            node_coords: [batch_size, num_nodes, 6]
            edge_index: [2, num_edges]
            edge_distances: [num_edges, 32]
            temperature: Sampling temperature
            use_cache: Whether to use KV caching (True for speedup)

        Returns:
            sequences: [batch_size, num_nodes]
        """
        # Encode structure
        encoder_output = self.encoder(node_coords, edge_index, edge_distances)

        # Add batch dimension if needed
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(0)

        # Generate sequence with caching
        sequences = self.decoder.generate_sequence(encoder_output, temperature, use_cache)

        return sequences


if __name__ == "__main__":
    print("Testing KV Cached ProteinMPNN...")

    model = KVCachedProteinMPNN(
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=8
    )

    # Test with and without cache
    batch_size = 2
    seq_len = 100
    num_edges = seq_len * 30

    node_coords = torch.randn(batch_size, seq_len, 6)
    edge_index = torch.randint(0, seq_len, (2, num_edges))
    edge_distances = torch.randn(num_edges, 32)

    print(f"Input shape: {node_coords.shape}")

    import time

    # Without cache
    start = time.time()
    sequences_no_cache = model(node_coords, edge_index, edge_distances, use_cache=False)
    time_no_cache = time.time() - start

    # With cache
    start = time.time()
    sequences_with_cache = model(node_coords, edge_index, edge_distances, use_cache=True)
    time_with_cache = time.time() - start

    print(f"\nWithout cache: {time_no_cache:.3f}s")
    print(f"With cache: {time_with_cache:.3f}s")
    print(f"Speedup: {time_no_cache / time_with_cache:.2f}x")
    print(f"Output shape: {sequences_with_cache.shape}")
