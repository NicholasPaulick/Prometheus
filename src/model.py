"""
Titans Model Interface

Unified interface for all Titans architecture variants:
- MAC (Memory-as-Context)
- MAG (Memory-as-Gate)  
- MAL (Memory-as-Layer)

With support for Miras memory configurations:
- Standard (L2 attentional bias)
- Yaad (Huber loss, robust to outliers)
- Moneta (Lp norms, strict memory)
- Memora (Simplex constraint, probability memory)
"""

import torch
from torch import nn
from typing import Literal, Optional

from .mac import MemoryAsContextTransformer, MemoryAsContextLM
from .mag import MemoryAsGateTransformer, MemoryAsGateLM
from .mal import MemoryAsLayerTransformer, MemoryAsLayerLM


class TitansConfig:
    """
    Configuration for Titans models.
    
    Encapsulates all hyperparameters for easy model creation.
    """
    
    def __init__(
        self,
        # Model architecture
        architecture: Literal['mac', 'mag', 'mal'] = 'mac',
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        
        # Vocabulary (for LM)
        vocab_size: int = 32000,
        max_seq_len: int = 8192,
        
        # Segment/chunk settings
        segment_len: int = 64,
        
        # Memory configuration
        memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
        memory_hidden_dim: int = 256,
        memory_depth: int = 2,
        memory_lr: float = 0.01,
        memory_momentum: float = 0.9,
        
        # Architecture-specific
        num_persistent_tokens: int = 8,  # MAC
        window_size: int = 256,           # MAG
        num_memory_tokens: int = 32,      # MAL
        
        # Training
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        self.architecture = architecture
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.segment_len = segment_len
        self.memory_type = memory_type
        self.memory_hidden_dim = memory_hidden_dim
        self.memory_depth = memory_depth
        self.memory_lr = memory_lr
        self.memory_momentum = memory_momentum
        self.num_persistent_tokens = num_persistent_tokens
        self.window_size = window_size
        self.num_memory_tokens = num_memory_tokens
        self.dropout = dropout
        self.tie_weights = tie_weights
    
    def to_dict(self) -> dict:
        return vars(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TitansConfig':
        return cls(**d)


class TitansLM(nn.Module):
    """
    Unified Titans Language Model.
    
    Creates the appropriate architecture based on configuration.
    
    Example:
        >>> config = TitansConfig(architecture='mac', dim=256, num_layers=4)
        >>> model = TitansLM(config)
        >>> logits = model(input_ids)
    """
    
    def __init__(self, config: TitansConfig):
        super().__init__()
        
        self.config = config
        
        if config.architecture == 'mac':
            self.model = MemoryAsContextLM(
                vocab_size=config.vocab_size,
                dim=config.dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                segment_len=config.segment_len,
                max_seq_len=config.max_seq_len,
                num_persistent_tokens=config.num_persistent_tokens,
                memory_type=config.memory_type,
                memory_hidden_dim=config.memory_hidden_dim,
                memory_depth=config.memory_depth,
                dropout=config.dropout,
                tie_weights=config.tie_weights
            )
        elif config.architecture == 'mag':
            self.model = MemoryAsGateLM(
                vocab_size=config.vocab_size,
                dim=config.dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                segment_len=config.segment_len,
                max_seq_len=config.max_seq_len,
                window_size=config.window_size,
                memory_type=config.memory_type,
                memory_hidden_dim=config.memory_hidden_dim,
                memory_depth=config.memory_depth,
                dropout=config.dropout,
                tie_weights=config.tie_weights
            )
        elif config.architecture == 'mal':
            self.model = MemoryAsLayerLM(
                vocab_size=config.vocab_size,
                dim=config.dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                segment_len=config.segment_len,
                max_seq_len=config.max_seq_len,
                num_memory_tokens=config.num_memory_tokens,
                memory_type=config.memory_type,
                memory_hidden_dim=config.memory_hidden_dim,
                memory_depth=config.memory_depth,
                dropout=config.dropout,
                tie_weights=config.tie_weights
            )
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Token IDs (batch, seq_len)
            
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
        """
        return self.model(x)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling with this probability
            
        Returns:
            output_ids: Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = self.forward(input_ids)
                next_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    @property
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience factory functions

def create_mac_model(
    vocab_size: int = 32000,
    dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    segment_len: int = 64,
    memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
    **kwargs
) -> TitansLM:
    """Create a MAC (Memory-as-Context) model."""
    config = TitansConfig(
        architecture='mac',
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        segment_len=segment_len,
        memory_type=memory_type,
        **kwargs
    )
    return TitansLM(config)


def create_mag_model(
    vocab_size: int = 32000,
    dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    segment_len: int = 64,
    window_size: int = 256,
    memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
    **kwargs
) -> TitansLM:
    """Create a MAG (Memory-as-Gate) model."""
    config = TitansConfig(
        architecture='mag',
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        segment_len=segment_len,
        window_size=window_size,
        memory_type=memory_type,
        **kwargs
    )
    return TitansLM(config)


def create_mal_model(
    vocab_size: int = 32000,
    dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    segment_len: int = 64,
    num_memory_tokens: int = 32,
    memory_type: Literal['standard', 'yaad', 'moneta', 'memora'] = 'standard',
    **kwargs
) -> TitansLM:
    """Create a MAL (Memory-as-Layer) model."""
    config = TitansConfig(
        architecture='mal',
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        segment_len=segment_len,
        num_memory_tokens=num_memory_tokens,
        memory_type=memory_type,
        **kwargs
    )
    return TitansLM(config)


# Miras-specific convenience functions

def create_yaad_model(
    architecture: Literal['mac', 'mag', 'mal'] = 'mac',
    **kwargs
) -> TitansLM:
    """Create a model using Yaad memory (Huber loss, outlier robust)."""
    config = TitansConfig(architecture=architecture, memory_type='yaad', **kwargs)
    return TitansLM(config)


def create_moneta_model(
    architecture: Literal['mac', 'mag', 'mal'] = 'mac',
    **kwargs
) -> TitansLM:
    """Create a model using Moneta memory (Lp norms, strict memory)."""
    config = TitansConfig(architecture=architecture, memory_type='moneta', **kwargs)
    return TitansLM(config)


def create_memora_model(
    architecture: Literal['mac', 'mag', 'mal'] = 'mac',
    **kwargs
) -> TitansLM:
    """Create a model using Memora memory (simplex constraint)."""
    config = TitansConfig(architecture=architecture, memory_type='memora', **kwargs)
    return TitansLM(config)
