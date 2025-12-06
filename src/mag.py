"""
Memory-as-Gate (MAG) Architecture for Titans

The MAG architecture uses a gating mechanism to dynamically balance
contributions from short-term (attention) and long-term (neural) memory.

From "Titans: Learning to Memorize at Test Time" (2501.00663)
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple
from .memory import NeuralMemory, YaadMemory, MonetaMemory, MemoraMemory


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention for short-term memory.
    
    Only attends to a fixed window of past tokens, providing
    accurate but local dependency modeling.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sliding window attention.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            output: Attended tensor (batch, seq_len, dim)
        """
        b, n, d = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, n, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create sliding window mask
        # Each position can only attend to [i - window_size, i]
        mask = torch.ones(n, n, device=x.device, dtype=torch.bool)
        for i in range(n):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = False
        
        # Apply causal + window mask
        causal_mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        mask = mask | causal_mask
        
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.proj(out)
        
        return out


class MemoryGate(nn.Module):
    """
    Gating mechanism to balance short-term and long-term memory.
    
    Learns to dynamically weight the contributions based on input.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        
        # Gate network: takes [short_term, long_term, input] and outputs gate value
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        short_term: torch.Tensor,
        long_term: torch.Tensor,
        input_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated combination of short-term and long-term memory.
        
        Args:
            short_term: Short-term memory output (batch, seq, dim)
            long_term: Long-term memory output (batch, seq, dim)
            input_x: Original input (batch, seq, dim)
            
        Returns:
            output: Gated combination (batch, seq, dim)
            gate_values: Gate values for analysis (batch, seq, dim)
        """
        # Concatenate for gate computation
        combined = torch.cat([short_term, long_term, input_x], dim=-1)
        
        # Compute gate (0 = all short-term, 1 = all long-term)
        gate = self.gate_net(combined)
        
        # Gated combination
        output = (1 - gate) * short_term + gate * long_term
        
        return output, gate


class MemoryAsGateLayer(nn.Module):
    """
    Single Memory-as-Gate layer combining short-term and long-term memory.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        segment_len: int,
        window_size: int = 256,
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Short-term memory (sliding window attention)
        self.short_term = SlidingWindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        # Long-term memory (neural memory)
        if memory_type == 'yaad':
            self.long_term = YaadMemory(dim=dim, chunk_size=segment_len, 
                                        hidden_dim=memory_hidden_dim, num_layers=memory_depth)
        elif memory_type == 'moneta':
            self.long_term = MonetaMemory(dim=dim, chunk_size=segment_len,
                                          hidden_dim=memory_hidden_dim, num_layers=memory_depth)
        elif memory_type == 'memora':
            self.long_term = MemoraMemory(dim=dim, chunk_size=segment_len,
                                          hidden_dim=memory_hidden_dim, num_layers=memory_depth)
        else:
            self.long_term = NeuralMemory(dim=dim, chunk_size=segment_len, memory_type='mlp',
                                          hidden_dim=memory_hidden_dim, num_layers=memory_depth)
        
        # Memory gate
        self.gate = MemoryGate(dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Forward pass through MAG layer.
        
        Returns:
            output: Processed tensor
            memory_state: Updated memory state
            gate_values: Gate values for analysis
        """
        # Pre-norm
        x_norm = self.norm1(x)
        
        # Short-term memory (sliding window attention)
        short_out = self.short_term(x_norm)
        
        # Long-term memory (neural memory)
        long_out, memory_state, surprises = self.long_term(x_norm, memory_state)
        
        # Gated combination
        gated_out, gate_values = self.gate(short_out, long_out, x_norm)
        
        # Residual connection
        x = x + gated_out
        
        # Feed-forward with pre-norm and residual
        x = x + self.ffn(self.norm2(x))
        
        return x, memory_state, gate_values


class MemoryAsGateTransformer(nn.Module):
    """
    Memory-as-Gate (MAG) Transformer Architecture.
    
    Uses gating to dynamically balance short-term (attention) and
    long-term (neural memory) contributions at each layer.
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        segment_len: int,
        window_size: int = 256,
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MemoryAsGateLayer(
                dim=dim,
                num_heads=num_heads,
                segment_len=segment_len,
                window_size=window_size,
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
        return_gates: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through MAG architecture.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            return_gates: Whether to return gate values for analysis
            
        Returns:
            output: Processed tensor (batch, seq_len, dim)
            all_gates: Optional list of gate values per layer
        """
        all_gates = []
        memory_states = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            x, memory_states[i], gates = layer(x, memory_states[i])
            if return_gates:
                all_gates.append(gates)
        
        x = self.final_norm(x)
        
        if return_gates:
            return x, all_gates
        return x


class MemoryAsGateLM(nn.Module):
    """
    Complete Language Model using Memory-as-Gate architecture.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        segment_len: int,
        max_seq_len: int = 8192,
        window_size: int = 256,
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
        
        self.backbone = MemoryAsGateTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            segment_len=segment_len,
            window_size=window_size,
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
        h = self.backbone(h)
        
        return self.to_logits(h)
