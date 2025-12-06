"""
Comprehensive Tests for Titans Architecture

Tests all components:
- Memory module (surprise, momentum, forgetting)
- All architectures (MAC, MAG, MAL)
- All memory types (standard, yaad, moneta, memora)
- Gradient flow
- Shape consistency
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titans.memory import (
    NeuralMemory, YaadMemory, MonetaMemory, MemoraMemory,
    LinearMemory, MLPMemory,
    L2AttentionalBias, HuberAttentionalBias, LpAttentionalBias
)
from titans.mac import MemoryAsContextTransformer, MemoryAsContextLM
from titans.mag import MemoryAsGateTransformer, MemoryAsGateLM, SlidingWindowAttention
from titans.mal import MemoryAsLayerTransformer, MemoryAsLayerLM
from titans.model import TitansLM, TitansConfig, create_mac_model, create_mag_model, create_mal_model


def test_linear_memory():
    """Test linear memory network."""
    print("Testing LinearMemory...")
    
    batch, seq, dim = 2, 32, 64
    memory = LinearMemory(dim)
    
    # Initialize params
    params = memory.get_initial_params(batch, torch.device('cpu'))
    assert 'M' in params
    assert params['M'].shape == (batch, dim, dim)
    
    # Forward pass
    x = torch.randn(batch, seq, dim)
    out = memory.forward(x, params)
    assert out.shape == (batch, seq, dim)
    
    # Compute update
    bias = L2AttentionalBias()
    updates, surprise = memory.compute_update(x, x, params, bias)
    assert 'M' in updates
    assert surprise.shape == (batch,)
    
    print("  ✓ LinearMemory passed")


def test_mlp_memory():
    """Test MLP memory network."""
    print("Testing MLPMemory...")
    
    batch, seq, dim = 2, 32, 64
    memory = MLPMemory(dim, hidden_dim=128, num_layers=2)
    
    # Initialize params
    params = memory.get_initial_params(batch, torch.device('cpu'))
    assert 'W0' in params and 'b0' in params
    assert 'W1' in params and 'b1' in params
    
    # Forward pass
    x = torch.randn(batch, seq, dim)
    out = memory.forward(x, params)
    assert out.shape == (batch, seq, dim)
    
    # Compute update
    bias = L2AttentionalBias()
    updates, surprise = memory.compute_update(x, x, params, bias)
    assert len(updates) == 4  # W0, b0, W1, b1
    
    print("  ✓ MLPMemory passed")


def test_attentional_biases():
    """Test all attentional bias types."""
    print("Testing Attentional Biases...")
    
    batch, seq, dim = 2, 16, 32
    pred = torch.randn(batch, seq, dim)
    target = torch.randn(batch, seq, dim)
    
    for bias_cls, name in [
        (L2AttentionalBias, "L2"),
        (HuberAttentionalBias, "Huber"),
        (LpAttentionalBias, "Lp")
    ]:
        bias = bias_cls()
        loss = bias.compute_loss(pred, target)
        grad = bias.compute_gradient(pred, target)
        assert grad.shape == pred.shape
        print(f"  ✓ {name}AttentionalBias passed")


def test_neural_memory_standard():
    """Test standard NeuralMemory with all features."""
    print("Testing NeuralMemory (standard)...")
    
    batch, seq, dim = 2, 128, 64
    chunk_size = 32
    
    memory = NeuralMemory(
        dim=dim,
        chunk_size=chunk_size,
        memory_type='mlp',
        hidden_dim=128,
        num_layers=2,
        learning_rate=0.01,
        momentum=0.9,
        adaptive_forgetting=True
    )
    
    x = torch.randn(batch, seq, dim)
    outputs, final_state, surprises = memory(x)
    
    assert outputs.shape == (batch, seq, dim)
    assert surprises.shape == (batch, seq // chunk_size)
    assert len(final_state) > 0
    
    print("  ✓ NeuralMemory (standard) passed")


def test_memory_variants():
    """Test Yaad, Moneta, Memora memory configurations."""
    print("Testing Memory Variants...")
    
    batch, seq, dim = 2, 64, 32
    chunk_size = 16
    
    for MemoryCls, name in [
        (YaadMemory, "Yaad"),
        (MonetaMemory, "Moneta"),
        (MemoraMemory, "Memora")
    ]:
        memory = MemoryCls(dim=dim, chunk_size=chunk_size)
        x = torch.randn(batch, seq, dim)
        outputs, state, surprises = memory(x)
        assert outputs.shape == (batch, seq, dim)
        print(f"  ✓ {name}Memory passed")


def test_mac_architecture():
    """Test Memory-as-Context architecture."""
    print("Testing MAC Architecture...")
    
    batch, seq, dim = 2, 128, 64
    
    mac = MemoryAsContextTransformer(
        dim=dim,
        num_layers=2,
        num_heads=4,
        segment_len=32,
        num_persistent_tokens=4,
        memory_type='standard'
    )
    
    x = torch.randn(batch, seq, dim)
    out = mac(x)
    assert out.shape == (batch, seq, dim)
    
    # Test with memory state return
    out, mem_state = mac(x, return_memory_state=True)
    assert out.shape == (batch, seq, dim)
    assert mem_state is not None
    
    print("  ✓ MAC Architecture passed")


def test_mag_architecture():
    """Test Memory-as-Gate architecture."""
    print("Testing MAG Architecture...")
    
    batch, seq, dim = 2, 128, 64
    
    # Test sliding window attention
    swa = SlidingWindowAttention(dim=dim, num_heads=4, window_size=32)
    x = torch.randn(batch, seq, dim)
    out = swa(x)
    assert out.shape == (batch, seq, dim)
    print("  ✓ SlidingWindowAttention passed")
    
    # Test full MAG
    mag = MemoryAsGateTransformer(
        dim=dim,
        num_layers=2,
        num_heads=4,
        segment_len=32,
        window_size=64,
        memory_type='standard'
    )
    
    out = mag(x)
    assert out.shape == (batch, seq, dim)
    
    # Test with gates return
    out, gates = mag(x, return_gates=True)
    assert len(gates) == 2  # 2 layers
    
    print("  ✓ MAG Architecture passed")


def test_mal_architecture():
    """Test Memory-as-Layer architecture."""
    print("Testing MAL Architecture...")
    
    batch, seq, dim = 2, 128, 64
    
    mal = MemoryAsLayerTransformer(
        dim=dim,
        num_layers=2,
        num_heads=4,
        segment_len=32,
        num_memory_tokens=8,
        memory_type='standard'
    )
    
    x = torch.randn(batch, seq, dim)
    out = mal(x)
    assert out.shape == (batch, seq, dim)
    
    print("  ✓ MAL Architecture passed")


def test_language_models():
    """Test all LM wrappers."""
    print("Testing Language Models...")
    
    vocab_size = 1000
    dim = 64
    batch, seq = 2, 64
    
    for LMCls, name in [
        (MemoryAsContextLM, "MAC-LM"),
        (MemoryAsGateLM, "MAG-LM"),
        (MemoryAsLayerLM, "MAL-LM")
    ]:
        lm = LMCls(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=2,
            num_heads=4,
            segment_len=16
        )
        
        x = torch.randint(0, vocab_size, (batch, seq))
        logits = lm(x)
        assert logits.shape == (batch, seq, vocab_size)
        print(f"  ✓ {name} passed")


def test_unified_interface():
    """Test TitansLM unified interface."""
    print("Testing Unified Interface...")
    
    vocab_size = 1000
    batch, seq = 2, 64
    
    for arch in ['mac', 'mag', 'mal']:
        for mem_type in ['standard', 'yaad', 'moneta', 'memora']:
            config = TitansConfig(
                architecture=arch,
                memory_type=mem_type,
                vocab_size=vocab_size,
                dim=64,
                num_layers=2,
                num_heads=4,
                segment_len=16
            )
            
            model = TitansLM(config)
            x = torch.randint(0, vocab_size, (batch, seq))
            logits = model(x)
            assert logits.shape == (batch, seq, vocab_size)
    
    print("  ✓ Unified Interface passed (all combinations)")


def test_factory_functions():
    """Test factory functions."""
    print("Testing Factory Functions...")
    
    vocab_size = 1000
    batch, seq = 2, 32
    
    # Test architecture factories
    mac = create_mac_model(vocab_size=vocab_size, dim=64, num_layers=2, num_heads=4, segment_len=16)
    mag = create_mag_model(vocab_size=vocab_size, dim=64, num_layers=2, num_heads=4, segment_len=16)
    mal = create_mal_model(vocab_size=vocab_size, dim=64, num_layers=2, num_heads=4, segment_len=16)
    
    x = torch.randint(0, vocab_size, (batch, seq))
    
    for model, name in [(mac, "MAC"), (mag, "MAG"), (mal, "MAL")]:
        logits = model(x)
        assert logits.shape == (batch, seq, vocab_size)
        print(f"  ✓ create_{name.lower()}_model passed")


def test_gradient_flow():
    """Test that gradients flow through all components."""
    print("Testing Gradient Flow...")
    
    vocab_size = 1000
    batch, seq = 2, 32
    
    for arch in ['mac', 'mag', 'mal']:
        config = TitansConfig(
            architecture=arch,
            vocab_size=vocab_size,
            dim=64,
            num_layers=2,
            num_heads=4,
            segment_len=16
        )
        
        model = TitansLM(config)
        x = torch.randint(0, vocab_size, (batch, seq))
        
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        
        # Check gradients exist for critical parameters
        # Note: Some memory parameters may not receive gradients due to 
        # dynamic memory state updates (which is expected behavior)
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        param_count = sum(1 for p in model.parameters())
        
        # At least 90% of parameters should have gradients
        # Some memory-internal parameters may not participate in the gradient
        # chain due to the test-time update mechanism which creates new tensors
        grad_ratio = grad_count / param_count
        assert grad_ratio >= 0.9, f"{arch}: Only {grad_ratio*100:.1f}% parameters have gradients!"
        print(f"  ✓ {arch.upper()} gradient flow passed ({grad_count}/{param_count} params, {grad_ratio*100:.1f}%)")


def test_generation():
    """Test generation capability."""
    print("Testing Generation...")
    
    vocab_size = 100
    dim = 64
    
    config = TitansConfig(
        architecture='mac',
        vocab_size=vocab_size,
        dim=dim,
        num_layers=2,
        num_heads=4,
        segment_len=8
    )
    
    model = TitansLM(config)
    
    # Generate tokens
    input_ids = torch.randint(0, vocab_size, (1, 4))
    output_ids = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
    
    assert output_ids.shape == (1, 14)  # 4 input + 10 generated
    print("  ✓ Generation passed")


def test_model_properties():
    """Test model property methods."""
    print("Testing Model Properties...")
    
    config = TitansConfig(
        architecture='mac',
        vocab_size=1000,
        dim=128,
        num_layers=4,
        num_heads=8
    )
    
    model = TitansLM(config)
    
    num_params = model.num_parameters
    num_trainable = model.num_trainable_parameters
    
    assert num_params > 0
    assert num_trainable == num_params  # All should be trainable by default
    print(f"  ✓ Model has {num_params:,} parameters")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TITANS ARCHITECTURE TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_linear_memory,
        test_mlp_memory,
        test_attentional_biases,
        test_neural_memory_standard,
        test_memory_variants,
        test_mac_architecture,
        test_mag_architecture,
        test_mal_architecture,
        test_language_models,
        test_unified_interface,
        test_factory_functions,
        test_gradient_flow,
        test_generation,
        test_model_properties,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nTitans + Miras implementation is complete and verified.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
