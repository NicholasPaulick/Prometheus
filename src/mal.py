"""
Memory-as-Layer (MAL) Architecture for Titans

The MAL architecture treats neural memory as a separate processing layer
that compresses context before it reaches the attention mechanism.

From "Titans: Learning to Memorize at Test Time" (2501.00663)
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple
from .memory import NeuralMemory, YaadMemory, MonetaMemory, MemoraMemory


class MemoryCompressionLayer(nn.Module):
    """
    Memory layer that compresses past and current context.
    
    This layer uses neural memory to create a compressed representation
    of the sequence that captures long-range dependencies.
    """
    
    def __init__(
        self,
        dim: int,
        segment_len: int,
        num_memory_tokens: int = 32,
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2
    ):
        super().__init__()
        
        self.dim = dim
        self.segment_len = segment_len
        self.num_memory_tokens = num_memory_tokens
        
        # Neural memory for compression
        if memory_type == 'yaad':
            self.memory = YaadMemory(dim=dim, chunk_size=segment_len,
                                     hidden_dim=memory_hidden_dim, num_layers=memory_depth)
        elif memory_type == 'moneta':
            self.memory = MonetaMemory(dim=dim, chunk_size=segment_len,
                                       hidden_dim=memory_hidden_dim, num_layers=memory_depth)
        elif memory_type == 'memora':
            self.memory = MemoraMemory(dim=dim, chunk_size=segment_len,
                                       hidden_dim=memory_hidden_dim, num_layers=memory_depth)
        else:
            self.memory = NeuralMemory(dim=dim, chunk_size=segment_len, memory_type='mlp',
                                       hidden_dim=memory_hidden_dim, num_layers=memory_depth)
        
        # Learnable memory queries for compression
        self.memory_queries = nn.Parameter(torch.randn(1, num_memory_tokens, dim) * 0.02)
        
        # Cross-attention for compression
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Projection for combining compressed memory with input
        self.combine_proj = nn.Linear(dim * 2, dim)
    
    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Compress input using neural memory.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_state: Optional previous memory state
            
        Returns:
            output: Compressed representation (batch, seq_len, dim)
            memory_state: Updated memory state
            compressed: Compressed memory tokens (batch, num_memory_tokens, dim)
        """
        b, n, d = x.shape
        
        # 1. Retrieve from neural memory
        memory_out, memory_state, surprises = self.memory(x, memory_state)
        
        # 2. Compress using cross-attention with learnable queries
        queries = self.memory_queries.expand(b, -1, -1)  # (b, num_tokens, d)
        
        # Keys and values from memory output
        compressed, _ = self.compress_attn(
            query=queries,
            key=memory_out,
            value=memory_out
        )
        compressed = self.norm(compressed)
        
        # 3. Broadcast compressed memory to all positions
        # Average the compressed tokens and expand
        compressed_summary = compressed.mean(dim=1, keepdim=True)  # (b, 1, d)
        compressed_expanded = compressed_summary.expand(-1, n, -1)  # (b, n, d)
        
        # 4. Combine with original input
        combined = torch.cat([x, compressed_expanded], dim=-1)
        output = self.combine_proj(combined)
        
        return output, memory_state, compressed


class MemoryAsLayerBlock(nn.Module):
    """
    A block that combines memory compression with standard attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        segment_len: int,
        num_memory_tokens: int = 32,
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Memory compression layer
        self.memory_layer = MemoryCompressionLayer(
            dim=dim,
            segment_len=segment_len,
            num_memory_tokens=num_memory_tokens,
            memory_type=memory_type,
            memory_hidden_dim=memory_hidden_dim,
            memory_depth=memory_depth
        )
        
        # Standard self-attention (operates on compressed representation)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[dict] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through MAL block.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_state: Optional previous memory state
            attn_mask: Optional attention mask
            
        Returns:
            output: Processed tensor (batch, seq_len, dim)
            memory_state: Updated memory state
        """
        # 1. Memory compression (acts as a layer before attention)
        compressed_x, memory_state, _ = self.memory_layer(x, memory_state)
        
        # 2. Self-attention with residual
        x_norm = self.norm1(compressed_x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = compressed_x + self.dropout(attn_out)
        
        # 3. Feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        
        return x, memory_state


class MemoryAsLayerTransformer(nn.Module):
    """
    Memory-as-Layer (MAL) Transformer Architecture.
    
    Each layer first compresses the input through neural memory,
    then applies standard self-attention on the compressed representation.
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        segment_len: int,
        num_memory_tokens: int = 32,
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MemoryAsLayerBlock(
                dim=dim,
                num_heads=num_heads,
                segment_len=segment_len,
                num_memory_tokens=num_memory_tokens,
                memory_type=memory_type,
                memory_hidden_dim=memory_hidden_dim,
                memory_depth=memory_depth,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through MAL architecture.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            attn_mask: Optional attention mask
            
        Returns:
            output: Processed tensor (batch, seq_len, dim)
        """
        memory_states = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            x, memory_states[i] = layer(x, memory_states[i], attn_mask)
        
        return self.final_norm(x)


class MemoryAsLayerLM(nn.Module):
    """
    Complete Language Model using Memory-as-Layer architecture.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        segment_len: int,
        max_seq_len: int = 8192,
        num_memory_tokens: int = 32,
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.backbone = MemoryAsLayerTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            segment_len=segment_len,
            num_memory_tokens=num_memory_tokens,
            memory_type=memory_type,
            memory_hidden_dim=memory_hidden_dim,
            memory_depth=memory_depth,
            dropout=dropout
        )
        
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        
        if tie_weights:
            self.to_logits.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape
        device = x.device
        
        tok_emb = self.token_embedding(x)
        pos = torch.arange(n, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        
        h = self.dropout(tok_emb + pos_emb)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(n, n, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        h = self.backbone(h, attn_mask=causal_mask)
        
        return self.to_logits(h)
