"""
Memory-as-Context (MAC) Architecture for Titans

The MAC architecture prepends memory tokens to the input sequence,
allowing the attention mechanism to attend to both current context
and retrieved long-term memory.

From "Titans: Learning to Memorize at Test Time" (2501.00663)
"""

import torch
from torch import nn
from typing import Optional, Literal
from .memory import NeuralMemory, YaadMemory, MonetaMemory, MemoraMemory


class PersistentMemory(nn.Module):
    """
    Persistent Memory Module - learnable, data-independent parameters.
    
    Unlike long-term memory which adapts at test time, persistent memory
    remains fixed and encodes task-specific knowledge learned during training.
    """
    
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Return persistent memory tokens expanded for batch."""
        return self.tokens.expand(batch_size, -1, -1)


class MemoryAsContextTransformer(nn.Module):
    """
    Memory-as-Context (MAC) Transformer Architecture.
    
    The MAC architecture:
    1. Processes input in segments
    2. Uses neural memory to retrieve relevant past context
    3. Prepends memory tokens to each segment
    4. Applies attention over [persistent, memory, input] tokens
    5. Only keeps the output for input tokens
    
    Args:
        dim: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        segment_len: Length of each segment for chunked processing
        num_persistent_tokens: Number of persistent memory tokens
        memory_type: Type of neural memory ('standard', 'yaad', 'moneta', 'memora')
        memory_config: Additional configuration for the memory module
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        segment_len: int,
        num_persistent_tokens: int = 8,
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2,
        memory_lr: float = 0.01,
        memory_momentum: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.segment_len = segment_len
        self.num_persistent = num_persistent_tokens
        
        # Persistent Memory (task-specific knowledge)
        self.persistent_memory = PersistentMemory(num_persistent_tokens, dim)
        
        # Neural Long-Term Memory
        if memory_type == 'yaad':
            self.memory = YaadMemory(
                dim=dim,
                chunk_size=segment_len,
                hidden_dim=memory_hidden_dim,
                num_layers=memory_depth,
                learning_rate=memory_lr,
                momentum=memory_momentum
            )
        elif memory_type == 'moneta':
            self.memory = MonetaMemory(
                dim=dim,
                chunk_size=segment_len,
                hidden_dim=memory_hidden_dim,
                num_layers=memory_depth,
                learning_rate=memory_lr,
                momentum=memory_momentum
            )
        elif memory_type == 'memora':
            self.memory = MemoraMemory(
                dim=dim,
                chunk_size=segment_len,
                hidden_dim=memory_hidden_dim,
                num_layers=memory_depth,
                learning_rate=memory_lr,
                momentum=memory_momentum
            )
        else:
            self.memory = NeuralMemory(
                dim=dim,
                chunk_size=segment_len,
                memory_type='mlp',
                hidden_dim=memory_hidden_dim,
                num_layers=memory_depth,
                learning_rate=memory_lr,
                momentum=memory_momentum
            )
        
        # Transformer layers with pre-norm
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
        
        # Projection for memory context (optional refinement)
        self.memory_proj = nn.Linear(dim, dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_memory_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through MAC architecture.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            return_memory_state: Whether to return the final memory state
            
        Returns:
            output: Processed tensor (batch, seq_len, dim)
            memory_state: Optional final memory state if return_memory_state=True
        """
        b, n, d = x.shape
        
        # 1. Get long-term memory retrieval
        memory_out, memory_state, surprises = self.memory(x)
        memory_out = self.memory_proj(memory_out)
        
        # 2. Pad for segment processing
        if n % self.segment_len != 0:
            padding = self.segment_len - (n % self.segment_len)
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            memory_out = torch.nn.functional.pad(memory_out, (0, 0, 0, padding))
        
        padded_len = x.shape[1]
        num_segments = padded_len // self.segment_len
        
        # Reshape into segments
        x_segments = x.view(b, num_segments, self.segment_len, d)
        mem_segments = memory_out.view(b, num_segments, self.segment_len, d)
        
        # 3. Get persistent memory (same for all segments)
        persistent = self.persistent_memory(b)  # (b, num_persistent, d)
        
        # 4. Process each segment with [persistent, memory, input] context
        all_outputs = []
        
        for i in range(num_segments):
            segment = x_segments[:, i, :, :]      # (b, seg_len, d)
            mem_seg = mem_segments[:, i, :, :]    # (b, seg_len, d)
            
            # Concatenate: [persistent | memory | input]
            # Total context: num_persistent + seg_len + seg_len
            combined = torch.cat([persistent, mem_seg, segment], dim=1)
            
            # Apply transformer layers
            for layer in self.layers:
                combined = layer(combined)
            
            # Extract only the input tokens (last seg_len tokens)
            output_segment = combined[:, -self.segment_len:, :]
            all_outputs.append(output_segment)
        
        # 5. Concatenate and trim
        output = torch.cat(all_outputs, dim=1)
        output = self.final_norm(output)
        
        if output.shape[1] > n:
            output = output[:, :n, :]
        
        if return_memory_state:
            return output, memory_state
        return output


class MemoryAsContextLM(nn.Module):
    """
    Complete Language Model using Memory-as-Context architecture.
    
    Wraps MAC with token embeddings and output projection for language modeling.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        segment_len: int,
        max_seq_len: int = 8192,
        num_persistent_tokens: int = 8,
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # MAC backbone
        self.backbone = MemoryAsContextTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            segment_len=segment_len,
            num_persistent_tokens=num_persistent_tokens,
            memory_type=memory_type,
            memory_hidden_dim=memory_hidden_dim,
            memory_depth=memory_depth,
            dropout=dropout
        )
        
        # Output projection
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        
        # Optionally tie weights
        if tie_weights:
            self.to_logits.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_memory_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for language modeling.
        
        Args:
            x: Token IDs (batch, seq_len)
            return_memory_state: Whether to return memory state
            
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
        """
        b, n = x.shape
        device = x.device
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        pos = torch.arange(n, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        
        h = self.dropout(tok_emb + pos_emb)
        
        # Process through MAC
        if return_memory_state:
            h, memory_state = self.backbone(h, return_memory_state=True)
            logits = self.to_logits(h)
            return logits, memory_state
        else:
            h = self.backbone(h)
            logits = self.to_logits(h)
            return logits
