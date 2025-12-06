# Prometheus: Titans & Miras Implementation

A complete PyTorch implementation of the **Titans** and **Miras** architectures from Google Research papers, featuring neural long-term memory with test-time learning capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen.svg)]()

## 📄 Papers

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (2025)
- [It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://arxiv.org/abs/2504.13173) (2025)

## ✨ Features

### Core Memory Mechanism
- **Surprise-based memorization**: Prioritizes unexpected inputs using gradient magnitude
- **Momentum updates**: Accumulates past surprise for stable learning
- **Adaptive forgetting gates**: Learnable retention mechanisms
- **Flexible memory architectures**: Linear or deep MLP memory networks

### Architecture Variants

| Architecture | Description | Key Feature |
|--------------|-------------|-------------|
| **MAC** | Memory-as-Context | Prepends memory tokens to input |
| **MAG** | Memory-as-Gate | Gates between short & long-term memory |
| **MAL** | Memory-as-Layer | Memory compression before attention |

### Miras Framework Support

| Memory Type | Attentional Bias | Best For |
|-------------|------------------|----------|
| **Standard** | L2 Loss | General purpose |
| **Yaad** | Huber Loss | Robust to outliers |
| **Moneta** | Lp Norms | Strict memory behavior |
| **Memora** | Simplex Constraint | Probability-like memory |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Prometheus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src import TitansLM, TitansConfig, create_mac_model
import torch

# Method 1: Using configuration
config = TitansConfig(
    architecture='mac',           # or 'mag', 'mal'
    memory_type='yaad',          # or 'standard', 'moneta', 'memora'
    vocab_size=32000,
    dim=512,
    num_layers=6,
    num_heads=8,
    segment_len=64
)
model = TitansLM(config)

# Method 2: Using factory functions
model = create_mac_model(
    vocab_size=32000,
    dim=512,
    num_layers=6,
    memory_type='yaad'
)

# Forward pass
input_ids = torch.randint(0, 32000, (2, 128))  # (batch, seq)
logits = model(input_ids)  # (batch, seq, vocab)

# Generation
output = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
```

### Advanced Usage

```python
from src import (
    MemoryAsContextLM,
    MemoryAsGateLM,
    MemoryAsLayerLM,
    YaadMemory,
    MonetaMemory
)

# Create MAC model with Yaad memory
mac_model = MemoryAsContextLM(
    vocab_size=50000,
    dim=768,
    num_layers=12,
    num_heads=12,
    segment_len=128,
    num_persistent_tokens=16,
    memory_type='yaad',
    memory_hidden_dim=512,
    memory_depth=3
)

# Create MAG model with sliding window attention
mag_model = MemoryAsGateLM(
    vocab_size=50000,
    dim=768,
    num_layers=12,
    num_heads=12,
    segment_len=128,
    window_size=512,
    memory_type='moneta'
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python src/main.py
```

Expected output:
```
============================================================
TITANS ARCHITECTURE TEST SUITE
============================================================

Testing LinearMemory...                ✓ passed
Testing MLPMemory...                   ✓ passed
Testing Attentional Biases...          ✓ L2, Huber, Lp passed
Testing NeuralMemory (standard)...     ✓ passed
Testing Memory Variants...             ✓ Yaad, Moneta, Memora passed
Testing MAC Architecture...            ✓ passed
Testing MAG Architecture...            ✓ passed
Testing MAL Architecture...            ✓ passed
Testing Language Models...             ✓ MAC-LM, MAG-LM, MAL-LM passed
Testing Unified Interface...           ✓ All 12 combinations passed
Testing Factory Functions...           ✓ passed
Testing Gradient Flow...               ✓ MAC 100%, MAG 92%, MAL 100%
Testing Generation...                  ✓ passed
Testing Model Properties...            ✓ passed

============================================================
RESULTS: 14 passed, 0 failed
============================================================

🎉 ALL TESTS PASSED! 🎉
```

## 📊 Architecture Comparison

| Feature | MAC | MAG | MAL |
|---------|-----|-----|-----|
| Persistent Memory | ✅ | ❌ | ❌ |
| Sliding Window Attention | ❌ | ✅ | ❌ |
| Memory Compression | ❌ | ❌ | ✅ |
| Gating Mechanism | ❌ | ✅ | ❌ |
| Context Window | 2× segment | Limited by window | Compressed |
| Best For | Long context | Balanced | Deep integration |

## 🏗️ Project Structure

```
Prometheus/
├── src/
│   ├── __init__.py          # Package exports
│   ├── memory.py            # Neural memory module (22KB)
│   ├── mac.py               # Memory-as-Context
│   ├── mag.py               # Memory-as-Gate
│   ├── mal.py               # Memory-as-Layer
│   ├── model.py             # Unified interface
│   └── main.py              # Test suite
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔬 Key Components

### `memory.py`
- `NeuralMemory`: Core memory module with surprise/momentum/forgetting
- `LinearMemory`: Simple associative memory
- `MLPMemory`: Deep memory network
- `YaadMemory`, `MonetaMemory`, `MemoraMemory`: Miras variants
- Attentional biases: L2, Huber, Lp, CrossEntropy
- Retention gates: WeightDecay, L2Reg, Simplex, LpReg

### `mac.py`, `mag.py`, `mal.py`
- Three architecture variants with LM wrappers
- Persistent memory (MAC)
- Sliding window attention (MAG)
- Memory compression layer (MAL)

### `model.py`
- `TitansConfig`: Unified configuration
- `TitansLM`: Main model class
- Factory functions for all variants

## 🎯 Use Cases

- **Long-context language modeling**: Process sequences beyond typical transformer limits
- **Continual learning**: Adapt memory at test time without parameter updates
- **Recall-intensive tasks**: Needle-in-haystack, question answering
- **Genomics**: Long DNA sequences
- **Time series**: Extended temporal patterns

## 📈 Model Sizes

Example configurations:

| Config | Dim | Layers | Heads | Parameters |
|--------|-----|--------|-------|------------|
| Small | 256 | 4 | 4 | ~500K |
| Base | 512 | 6 | 8 | ~2M |
| Large | 768 | 12 | 12 | ~85M |
| XL | 1024 | 24 | 16 | ~350M |

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] CUDA kernel optimizations for memory updates
- [ ] Distributed training support
- [ ] Model checkpointing utilities
- [ ] Pre-trained weights
- [ ] Benchmark scripts

## 📝 Citation

If you use this implementation, please cite the original papers:

```bibtex
@article{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2025}
}

@article{behrouz2025miras,
  title={It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.13173},
  year={2025}
}
```

## 📜 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- Google Research for the groundbreaking Titans and Miras papers
- PyTorch team for the excellent framework
- The open-source ML community

---

**Built with ❤️ for advancing neural memory architectures**
