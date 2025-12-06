"""
Neural Memory Module for Titans Architecture

Implements the core long-term memory mechanism from:
- "Titans: Learning to Memorize at Test Time" (2501.00663)
- "It's All Connected: A Journey Through Test-Time Memorization..." (2504.13173)

Key Features:
- Surprise-based memorization (gradient as surprise metric)
- Momentum-based updates (accumulated past surprise)
- Adaptive forgetting gate (weight decay / retention regularization)
- Support for both Linear and MLP memory architectures
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
from abc import ABC, abstractmethod


# =============================================================================
# Attentional Bias Objectives (Miras Framework)
# =============================================================================

class AttentionalBias(ABC):
    """
    Abstract base class for attentional bias objectives.
    
    The attentional bias determines what the memory prioritizes learning.
    Different objectives lead to different memory behaviors.
    """
    
    @abstractmethod
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between prediction and target."""
        pass
    
    @abstractmethod
    def compute_gradient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the loss w.r.t. prediction (for surprise metric)."""
        pass


class L2AttentionalBias(AttentionalBias):
    """
    Standard L2 (MSE) attentional bias - used in original Titans.
    Loss = ||prediction - target||^2
    """
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target, reduction='none').sum(dim=-1).mean()
    
    def compute_gradient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Gradient of MSE: 2 * (prediction - target) / n
        return 2 * (prediction - target) / prediction.shape[-1]


class HuberAttentionalBias(AttentionalBias):
    """
    Huber loss attentional bias - used in Yaad model.
    More robust to outliers than L2.
    """
    
    def __init__(self, delta: float = 1.0):
        self.delta = delta
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(prediction, target, reduction='mean', delta=self.delta)
    
    def compute_gradient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        abs_diff = torch.abs(diff)
        # Huber gradient: diff if |diff| <= delta, else delta * sign(diff)
        grad = torch.where(abs_diff <= self.delta, diff, self.delta * torch.sign(diff))
        return grad / prediction.shape[-1]


class LpAttentionalBias(AttentionalBias):
    """
    Lp norm attentional bias - used in Moneta model.
    Generalized norm penalty for stricter memory behavior.
    """
    
    def __init__(self, p: float = 1.5):
        self.p = p
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        return (torch.abs(diff) ** self.p).sum(dim=-1).mean()
    
    def compute_gradient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        abs_diff = torch.abs(diff)
        # Gradient of |x|^p = p * |x|^(p-1) * sign(x)
        grad = self.p * (abs_diff ** (self.p - 1)) * torch.sign(diff)
        return grad / prediction.shape[-1]


class CrossEntropyAttentionalBias(AttentionalBias):
    """
    Cross-entropy attentional bias for probability-like memory states.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_logits = prediction / self.temperature
        target_probs = F.softmax(target / self.temperature, dim=-1)
        return F.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)),
            target_probs.view(-1, target_probs.size(-1)).argmax(dim=-1),
            reduction='mean'
        )
    
    def compute_gradient(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_probs = F.softmax(prediction / self.temperature, dim=-1)
        target_probs = F.softmax(target / self.temperature, dim=-1)
        return (pred_probs - target_probs) / self.temperature


# =============================================================================
# Retention Gates (Miras Framework)
# =============================================================================

class RetentionGate(ABC):
    """
    Abstract base class for retention gates (forgetting mechanisms).
    
    Retention gates control how much of the old memory is preserved
    vs. how much new information is incorporated.
    """
    
    @abstractmethod
    def apply(self, memory: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Apply retention/forgetting to memory state."""
        pass


class WeightDecayRetention(RetentionGate):
    """
    Standard weight decay retention - used in Titans.
    memory_new = (1 - alpha) * memory_old
    """
    
    def apply(self, memory: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return (1 - alpha) * memory


class L2RegularizationRetention(RetentionGate):
    """
    L2 regularization-based retention.
    Applies exponential decay based on memory magnitude.
    """
    
    def __init__(self, lambda_reg: float = 0.01):
        self.lambda_reg = lambda_reg
    
    def apply(self, memory: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        decay = torch.exp(-self.lambda_reg * alpha * (memory ** 2).sum(dim=-1, keepdim=True))
        return memory * decay


class SimplexRetention(RetentionGate):
    """
    Simplex projection retention - used in Memora.
    Projects memory onto probability simplex for stable updates.
    """
    
    def apply(self, memory: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # Apply decay first
        decayed = (1 - alpha) * memory
        # Project to simplex (normalize to sum to 1 along last dimension)
        # Using softmax as a differentiable approximation
        return F.softmax(decayed, dim=-1) * decayed.abs().sum(dim=-1, keepdim=True)


class LpRegularizationRetention(RetentionGate):
    """
    Lp regularization retention - used in Moneta.
    Generalized norm-based decay.
    """
    
    def __init__(self, p: float = 1.5, lambda_reg: float = 0.01):
        self.p = p
        self.lambda_reg = lambda_reg
    
    def apply(self, memory: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        norm = (torch.abs(memory) ** self.p).sum(dim=-1, keepdim=True) ** (1/self.p)
        decay = torch.exp(-self.lambda_reg * alpha * norm)
        return memory * decay


# =============================================================================
# Memory Architectures
# =============================================================================

class MemoryNetwork(nn.Module):
    """
    Base class for memory network architectures.
    Maps keys to values: f(key) -> value
    """
    
    @abstractmethod
    def forward(self, keys: torch.Tensor, params: dict) -> torch.Tensor:
        """Retrieve values for given keys using current parameters."""
        pass
    
    @abstractmethod
    def get_initial_params(self, batch_size: int, device: torch.device) -> dict:
        """Get initial memory parameters for a batch."""
        pass
    
    @abstractmethod
    def compute_update(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        params: dict,
        attentional_bias: AttentionalBias
    ) -> Tuple[dict, torch.Tensor]:
        """Compute parameter updates and surprise metric."""
        pass


class LinearMemory(MemoryNetwork):
    """
    Linear associative memory: M @ key = value
    The memory is a (dim_in, dim_out) matrix.
    """
    
    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
    
    def forward(self, keys: torch.Tensor, params: dict) -> torch.Tensor:
        # keys: (batch, seq, dim)
        # params['M']: (batch, dim, dim_out)
        return torch.bmm(keys, params['M'])
    
    def get_initial_params(self, batch_size: int, device: torch.device) -> dict:
        return {'M': torch.zeros(batch_size, self.dim, self.dim_out, device=device)}
    
    def compute_update(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        params: dict,
        attentional_bias: AttentionalBias
    ) -> Tuple[dict, torch.Tensor]:
        # Predict and compute error
        prediction = self.forward(keys, params)
        error_grad = attentional_bias.compute_gradient(prediction, values)
        
        # Gradient w.r.t. M: keys^T @ error_grad
        # keys: (batch, seq, dim), error_grad: (batch, seq, dim_out)
        update = torch.bmm(keys.transpose(1, 2), error_grad)
        
        # Surprise = magnitude of the gradient
        surprise = torch.norm(update, dim=(-2, -1))
        
        return {'M': update}, surprise


class MLPMemory(MemoryNetwork):
    """
    Multi-layer Perceptron memory network.
    Deeper memory for more complex associations.
    
    Architecture: key -> Linear -> ReLU -> ... -> Linear -> value
    """
    
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int = 256, 
        num_layers: int = 2,
        dim_out: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dim_out = dim_out or dim
        
        # Define layer dimensions
        self.layer_dims = [dim] + [hidden_dim] * (num_layers - 1) + [self.dim_out]
    
    def forward(self, keys: torch.Tensor, params: dict) -> torch.Tensor:
        x = keys
        for i in range(self.num_layers):
            W = params[f'W{i}']  # (batch, in_dim, out_dim)
            b = params[f'b{i}']  # (batch, 1, out_dim)
            x = torch.bmm(x, W) + b
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x
    
    def get_initial_params(self, batch_size: int, device: torch.device) -> dict:
        params = {}
        for i in range(self.num_layers):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]
            # Xavier initialization
            std = (2.0 / (in_dim + out_dim)) ** 0.5
            params[f'W{i}'] = torch.randn(batch_size, in_dim, out_dim, device=device) * std
            params[f'b{i}'] = torch.zeros(batch_size, 1, out_dim, device=device)
        return params
    
    def compute_update(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        params: dict,
        attentional_bias: AttentionalBias
    ) -> Tuple[dict, torch.Tensor]:
        """
        Compute updates using backpropagation through the MLP.
        """
        # Forward pass with intermediate activations
        # We store the INPUT to each layer (after previous activation)
        layer_inputs = [keys]  # Input to layer 0
        pre_activations = []   # Pre-ReLU values for gradient computation
        
        x = keys
        for i in range(self.num_layers):
            W = params[f'W{i}']
            b = params[f'b{i}']
            z = torch.bmm(x, W) + b  # Pre-activation
            
            if i < self.num_layers - 1:
                pre_activations.append(z)
                x = F.relu(z)
                layer_inputs.append(x)  # Input to next layer
            else:
                x = z  # No activation on last layer
        
        prediction = x
        
        # Backward pass
        error_grad = attentional_bias.compute_gradient(prediction, values)
        
        updates = {}
        total_surprise = 0.0
        
        for i in range(self.num_layers - 1, -1, -1):
            W = params[f'W{i}']
            
            # Input to this layer
            layer_input = layer_inputs[i]
            
            # Gradient w.r.t. W: layer_input^T @ error_grad
            dW = torch.bmm(layer_input.transpose(1, 2), error_grad)
            # Gradient w.r.t. b: sum over sequence dimension
            db = error_grad.sum(dim=1, keepdim=True)
            
            updates[f'W{i}'] = dW
            updates[f'b{i}'] = db
            
            total_surprise = total_surprise + torch.norm(dW, dim=(-2, -1))
            
            if i > 0:
                # Backprop through linear
                error_grad = torch.bmm(error_grad, W.transpose(1, 2))
                # Backprop through ReLU (use pre_activation from previous layer)
                pre_act = pre_activations[i - 1]
                error_grad = error_grad * (pre_act > 0).float()
        
        return updates, total_surprise


# =============================================================================
# Main Neural Memory Module
# =============================================================================

class NeuralMemory(nn.Module):
    """
    Neural Long-Term Memory Module for Titans Architecture.
    
    Implements test-time memorization with:
    - Surprise-based learning (prioritize unexpected inputs)
    - Momentum-based updates (accumulated past surprise)
    - Adaptive forgetting (weight decay / retention gates)
    
    Args:
        dim: Input/output dimension
        chunk_size: Size of chunks for processing
        memory_type: 'linear' or 'mlp'
        hidden_dim: Hidden dimension for MLP memory
        num_layers: Number of layers for MLP memory
        learning_rate: Step size for memory updates (sigma)
        momentum: Momentum coefficient (beta)
        forgetting_factor: Base forgetting factor (alpha)
        adaptive_forgetting: Whether to use learned forgetting gates
        attentional_bias: Type of attentional bias ('l2', 'huber', 'lp', 'cross_entropy')
        retention_gate: Type of retention gate ('weight_decay', 'l2_reg', 'simplex', 'lp_reg')
    """
    
    def __init__(
        self,
        dim: int,
        chunk_size: int = 32,
        memory_type: Literal['linear', 'mlp'] = 'linear',
        hidden_dim: int = 256,
        num_layers: int = 2,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        forgetting_factor: float = 0.01,
        adaptive_forgetting: bool = True,
        attentional_bias: Literal['l2', 'huber', 'lp', 'cross_entropy'] = 'l2',
        retention_gate: Literal['weight_decay', 'l2_reg', 'simplex', 'lp_reg'] = 'weight_decay'
    ):
        super().__init__()
        
        self.dim = dim
        self.chunk_size = chunk_size
        self.learning_rate = learning_rate
        self.momentum_coef = momentum
        self.base_forgetting = forgetting_factor
        self.adaptive_forgetting = adaptive_forgetting
        
        # Initialize memory network
        if memory_type == 'linear':
            self.memory_net = LinearMemory(dim)
        else:
            self.memory_net = MLPMemory(dim, hidden_dim, num_layers)
        
        # Initialize attentional bias
        self.attentional_bias = self._create_attentional_bias(attentional_bias)
        
        # Initialize retention gate
        self.retention = self._create_retention_gate(retention_gate)
        
        # Learnable forgetting gate (if adaptive)
        if adaptive_forgetting:
            self.forget_gate = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )
        
        # Learnable surprise scaling
        self.surprise_scale = nn.Parameter(torch.ones(1))
    
    def _create_attentional_bias(self, bias_type: str) -> AttentionalBias:
        if bias_type == 'l2':
            return L2AttentionalBias()
        elif bias_type == 'huber':
            return HuberAttentionalBias()
        elif bias_type == 'lp':
            return LpAttentionalBias()
        elif bias_type == 'cross_entropy':
            return CrossEntropyAttentionalBias()
        else:
            raise ValueError(f"Unknown attentional bias: {bias_type}")
    
    def _create_retention_gate(self, gate_type: str) -> RetentionGate:
        if gate_type == 'weight_decay':
            return WeightDecayRetention()
        elif gate_type == 'l2_reg':
            return L2RegularizationRetention()
        elif gate_type == 'simplex':
            return SimplexRetention()
        elif gate_type == 'lp_reg':
            return LpRegularizationRetention()
        else:
            raise ValueError(f"Unknown retention gate: {gate_type}")
    
    def _apply_momentum_update(
        self,
        params: dict,
        updates: dict,
        momentum_state: dict,
        alpha: torch.Tensor
    ) -> Tuple[dict, dict]:
        """Apply momentum-based update with forgetting."""
        new_params = {}
        new_momentum = {}
        
        for key in params:
            # Update momentum: m = beta * m_prev + gradient
            if key in momentum_state:
                new_m = self.momentum_coef * momentum_state[key] + updates[key]
            else:
                new_m = updates[key]
            new_momentum[key] = new_m
            
            # Apply retention/forgetting
            retained = self.retention.apply(params[key], alpha)
            
            # Update parameters: theta = retained - lr * momentum
            new_params[key] = retained - self.learning_rate * self.surprise_scale * new_m
        
        return new_params, new_momentum
    
    def forward(
        self, 
        x: torch.Tensor,
        initial_memory: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Process input sequence through neural memory.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            initial_memory: Optional initial memory state
            
        Returns:
            outputs: Retrieved memory for each position (batch, seq_len, dim)
            final_memory: Final memory state (dict of tensors)
            surprises: Surprise values per chunk (batch, num_chunks)
        """
        b, n, d = x.shape
        device = x.device
        
        # Pad sequence if necessary
        orig_len = n
        if n % self.chunk_size != 0:
            padding = self.chunk_size - (n % self.chunk_size)
            x = F.pad(x, (0, 0, 0, padding))
            n = x.shape[1]
        
        # Reshape into chunks
        num_chunks = n // self.chunk_size
        x_chunks = x.view(b, num_chunks, self.chunk_size, d)
        
        # Initialize memory state
        if initial_memory is None:
            params = self.memory_net.get_initial_params(b, device)
        else:
            params = initial_memory
        
        momentum_state = {}
        outputs = []
        surprises = []
        
        # Process chunks sequentially
        for i in range(num_chunks):
            chunk = x_chunks[:, i, :, :]  # (batch, chunk_size, dim)
            
            # 1. RETRIEVE: Use current memory to predict
            retrieved = self.memory_net.forward(chunk, params)
            outputs.append(retrieved)
            
            # 2. COMPUTE SURPRISE: How unexpected is this chunk?
            updates, surprise = self.memory_net.compute_update(
                chunk, chunk, params, self.attentional_bias
            )
            surprises.append(surprise)
            
            # 3. COMPUTE FORGETTING FACTOR
            if self.adaptive_forgetting:
                # Use chunk mean to compute adaptive forgetting
                chunk_summary = chunk.mean(dim=1)  # (batch, dim)
                alpha = self.forget_gate(chunk_summary)  # (batch, 1)
                alpha = alpha.unsqueeze(-1)  # (batch, 1, 1) for broadcasting
            else:
                alpha = torch.full((b, 1, 1), self.base_forgetting, device=device)
            
            # 4. MEMORIZE: Update memory with momentum and forgetting
            params, momentum_state = self._apply_momentum_update(
                params, updates, momentum_state, alpha
            )
        
        # Concatenate outputs
        outputs = torch.cat(outputs, dim=1)
        surprises = torch.stack(surprises, dim=1)  # (batch, num_chunks)
        
        # Remove padding
        if outputs.shape[1] > orig_len:
            outputs = outputs[:, :orig_len, :]
        
        return outputs, params, surprises


# =============================================================================
# Specialized Memory Configurations (Miras Models)
# =============================================================================

class YaadMemory(NeuralMemory):
    """
    Yaad memory configuration from Miras paper.
    Uses Huber loss for robustness to outliers.
    Named after Persian word for "memory".
    """
    
    def __init__(self, dim: int, chunk_size: int = 32, **kwargs):
        super().__init__(
            dim=dim,
            chunk_size=chunk_size,
            memory_type='mlp',
            attentional_bias='huber',
            retention_gate='weight_decay',
            **kwargs
        )


class MonetaMemory(NeuralMemory):
    """
    Moneta memory configuration from Miras paper.
    Uses Lp norms for stricter memory behavior.
    """
    
    def __init__(self, dim: int, chunk_size: int = 32, **kwargs):
        super().__init__(
            dim=dim,
            chunk_size=chunk_size,
            memory_type='mlp',
            attentional_bias='lp',
            retention_gate='lp_reg',
            **kwargs
        )


class MemoraMemory(NeuralMemory):
    """
    Memora memory configuration from Miras paper.
    Uses simplex constraints for probability-like memory states.
    """
    
    def __init__(self, dim: int, chunk_size: int = 32, **kwargs):
        super().__init__(
            dim=dim,
            chunk_size=chunk_size,
            memory_type='mlp',
            attentional_bias='cross_entropy',
            retention_gate='simplex',
            **kwargs
        )
