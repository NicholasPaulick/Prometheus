"""
Titans Package

A PyTorch implementation of the Titans architecture from:
- "Titans: Learning to Memorize at Test Time" (2501.00663)
- "It's All Connected: A Journey Through Test-Time Memorization..." (2504.13173)

Architectures:
- MAC: Memory-as-Context (prepends memory tokens to input)
- MAG: Memory-as-Gate (gates between short and long-term memory)
- MAL: Memory-as-Layer (uses memory for context compression)

Memory Types (from Miras framework):
- standard: L2 attentional bias (original Titans)
- yaad: Huber loss (robust to outliers)
- moneta: Lp norms (strict memory behavior)
- memora: Simplex constraint (probability memory)

Example:
    >>> from src import TitansLM, TitansConfig
    >>> 
    >>> # Create a MAC model with Yaad memory
    >>> config = TitansConfig(
    ...     architecture='mac',
    ...     memory_type='yaad',
    ...     dim=256,
    ...     num_layers=4
    ... )
    >>> model = TitansLM(config)
    >>> 
    >>> # Or use factory functions
    >>> from src import create_mac_model, create_yaad_model
    >>> model = create_mac_model(dim=256, num_layers=4)
    >>> model = create_yaad_model(architecture='mag', dim=512)
"""

# Core memory module
from .memory import (
    NeuralMemory,
    YaadMemory,
    MonetaMemory,
    MemoraMemory,
    # Attentional biases
    AttentionalBias,
    L2AttentionalBias,
    HuberAttentionalBias,
    LpAttentionalBias,
    CrossEntropyAttentionalBias,
    # Retention gates
    RetentionGate,
    WeightDecayRetention,
    L2RegularizationRetention,
    SimplexRetention,
    LpRegularizationRetention,
    # Memory networks
    MemoryNetwork,
    LinearMemory,
    MLPMemory,
)

# Architecture modules
from .mac import (
    MemoryAsContextTransformer,
    MemoryAsContextLM,
    PersistentMemory,
)

from .mag import (
    MemoryAsGateTransformer,
    MemoryAsGateLM,
    SlidingWindowAttention,
    MemoryGate,
)

from .mal import (
    MemoryAsLayerTransformer,
    MemoryAsLayerLM,
    MemoryCompressionLayer,
)

# Unified interface
from .model import (
    TitansConfig,
    TitansLM,
    create_mac_model,
    create_mag_model,
    create_mal_model,
    create_yaad_model,
    create_moneta_model,
    create_memora_model,
)

__version__ = "1.0.0"
__author__ = "Titans Implementation"

__all__ = [
    # Config and main model
    "TitansConfig",
    "TitansLM",
    
    # Factory functions
    "create_mac_model",
    "create_mag_model",
    "create_mal_model",
    "create_yaad_model",
    "create_moneta_model",
    "create_memora_model",
    
    # Memory modules
    "NeuralMemory",
    "YaadMemory",
    "MonetaMemory",
    "MemoraMemory",
    
    # Attentional biases
    "AttentionalBias",
    "L2AttentionalBias",
    "HuberAttentionalBias",
    "LpAttentionalBias",
    "CrossEntropyAttentionalBias",
    
    # Retention gates
    "RetentionGate",
    "WeightDecayRetention",
    "L2RegularizationRetention",
    "SimplexRetention",
    "LpRegularizationRetention",
    
    # Memory networks
    "MemoryNetwork",
    "LinearMemory",
    "MLPMemory",
    
    # MAC
    "MemoryAsContextTransformer",
    "MemoryAsContextLM",
    "PersistentMemory",
    
    # MAG
    "MemoryAsGateTransformer",
    "MemoryAsGateLM",
    "SlidingWindowAttention",
    "MemoryGate",
    
    # MAL
    "MemoryAsLayerTransformer",
    "MemoryAsLayerLM",
    "MemoryCompressionLayer",
]
