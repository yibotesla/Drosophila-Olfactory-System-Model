"""
DrosophilaOlfactoryModel - Main model class for the Drosophila olfactory system.

This module implements the complete three-layer feedforward network:
- Input layer (AL/PN): Receives odor vectors
- Sparse encoding layer (MB/KC): Implements kWTA sparse coding
- Output layer (MBON): Produces behavioral output

Includes modulation via DANs for associative learning using three-factor rule.
"""

import json
import time
from typing import Tuple, Optional, List, TypedDict, TYPE_CHECKING
import numpy as np

from src.sparse_encoder import SparseEncoder

if TYPE_CHECKING:
    from src.config import ModelConfig


class LearningEvent(TypedDict):
    """学习事件记录结构"""
    type: str           # 'aversive' | 'appetitive'
    odor_hash: int      # 气味向量哈希
    strength: float     # 刺激强度
    weight_change: float  # 权重变化量
    timestamp: float    # 时间戳


class DrosophilaOlfactoryModel:
    """
    果蝇嗅觉系统模型 - Drosophila Olfactory System Model
    
    Implements a biologically-inspired model of the Drosophila olfactory circuit
    with sparse coding in the mushroom body and three-factor learning rule.
    """
    
    def __init__(
        self,
        n_pn: int = 50,
        n_kc: int = 2000,
        n_mbon: int = 1,
        sparsity: float = 0.05,
        learning_rate: float = 0.05,
        connectivity: float = 0.14,
        seed: Optional[int] = None
    ):
        """
        Initialize the Drosophila Olfactory Model.
        
        Args:
            n_pn: Number of projection neurons (input dimension). Default 50.
            n_kc: Number of Kenyon cells. Default 2000.
            n_mbon: Number of mushroom body output neurons. Default 1.
            sparsity: Fraction of KCs to keep active (kWTA). Default 0.05 (5%).
            learning_rate: Learning rate (η) for weight updates. Default 0.05.
            connectivity: Fraction of PNs each KC connects to. Default 0.14.
            seed: Optional random seed for reproducibility.
            
        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        # Validate parameters
        if n_pn <= 0:
            raise ValueError(f"n_pn must be positive, got {n_pn}")
        if n_kc <= 0:
            raise ValueError(f"n_kc must be positive, got {n_kc}")
        if n_mbon <= 0:
            raise ValueError(f"n_mbon must be positive, got {n_mbon}")
        if not (0 < sparsity < 1):
            raise ValueError(f"sparsity must be in (0, 1), got {sparsity}")
        if learning_rate < 0:
            raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")
        if not (0 < connectivity <= 1):
            raise ValueError(f"connectivity must be in (0, 1], got {connectivity}")

        # Store parameters
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.n_mbon = n_mbon
        self.sparsity = sparsity
        self.learning_rate = learning_rate
        self.connectivity = connectivity
        self._seed = seed
        
        # Initialize sparse encoder (PN -> KC layer with kWTA)
        self._encoder = SparseEncoder(
            n_input=n_pn,
            n_output=n_kc,
            sparsity=sparsity,
            connectivity=connectivity,
            seed=seed
        )
        
        # Initialize KC -> MBON weight matrix (plastic, all weights start at 1.0)
        # Shape: (n_kc, n_mbon)
        self._W_kc_mbon = np.ones((n_kc, n_mbon), dtype=np.float64)
        
        # Store initial weights for reset functionality
        self._W_kc_mbon_initial = self._W_kc_mbon.copy()
        
        # Initialize learning history (Requirements 5.1)
        self._learning_history: List[LearningEvent] = []
    
    @classmethod
    def from_config(cls, config: 'ModelConfig') -> 'DrosophilaOlfactoryModel':
        """
        Create a model instance from a ModelConfig object.
        
        Args:
            config: ModelConfig instance with all model parameters
            
        Returns:
            DrosophilaOlfactoryModel instance configured with the provided parameters
            
        Raises:
            ValueError: If config parameters are invalid
            
        Requirements: 7.4
        """
        # Validate config before creating model
        config.validate()
        
        return cls(
            n_pn=config.n_pn,
            n_kc=config.n_kc,
            n_mbon=config.n_mbon,
            sparsity=config.sparsity,
            learning_rate=config.learning_rate,
            connectivity=config.connectivity,
            seed=config.seed
        )
    
    def _validate_input(self, odor_input: np.ndarray, name: str = "odor") -> None:
        """
        验证输入向量的有效性。
        
        Args:
            odor_input: 待验证的输入向量
            name: 输入参数名称，用于错误信息
            
        Raises:
            TypeError: 如果输入不是 np.ndarray 类型
            ValueError: 如果输入维度不正确或包含 NaN/Inf 值
        """
        # 检查类型
        if not isinstance(odor_input, np.ndarray):
            raise TypeError(f"{name} must be np.ndarray, got {type(odor_input).__name__}")
        
        # 检查维度是否为一维
        if odor_input.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {odor_input.shape}")
        
        # 检查维度是否与 n_pn 匹配
        if odor_input.shape[0] != self.n_pn:
            raise ValueError(
                f"{name} dimension mismatch: expected {self.n_pn}, got {odor_input.shape[0]}"
            )
        
        # 检查 NaN 值
        if np.isnan(odor_input).any():
            raise ValueError(f"{name} contains NaN values")
        
        # 检查 Inf 值
        if np.isinf(odor_input).any():
            raise ValueError(f"{name} contains Inf values")
    
    @property
    def weights_kc_mbon(self) -> np.ndarray:
        """Get the KC-MBON weight matrix (read-only copy)."""
        return self._W_kc_mbon.copy()
    
    @property
    def weights_pn_kc(self) -> np.ndarray:
        """Get the PN-KC weight matrix (read-only copy)."""
        return self._encoder.weights
    
    def predict(self, odor_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict MBON output for a given odor input.
        
        Processes the odor through the PN->KC->MBON pathway:
        1. Sparse encoding via kWTA in KC layer
        2. Weighted sum to MBON output
        
        Args:
            odor_input: Odor vector of shape (n_pn,)
            
        Returns:
            Tuple of (mbon_output, kc_activation):
            - mbon_output: np.ndarray of shape (n_mbon,) with MBON responses
            - kc_activation: np.ndarray of shape (n_kc,) with binary KC activations
            
        Raises:
            TypeError: If input is not np.ndarray
            ValueError: If input dimension doesn't match n_pn, or contains NaN/Inf
        """
        # 验证输入
        self._validate_input(odor_input, "odor_input")
        
        # Step 1: Sparse encoding in KC layer
        kc_activation = self._encoder.encode(odor_input)
        
        # Step 2: Compute MBON output as weighted sum: y = Σ(w_i × KC_i)
        # kc_activation: (n_kc,), W_kc_mbon: (n_kc, n_mbon) -> output: (n_mbon,)
        mbon_output = kc_activation @ self._W_kc_mbon
        
        return mbon_output, kc_activation
    
    def modulate(self, kc_active: np.ndarray, modulatory_signal: float) -> None:
        """
        Apply modulatory signal to update KC-MBON weights using multiplicative learning rule.
        
        Multiplicative learning rules:
        - LTD (R > 0): w = w × (1 - η × R)
        - LTP (R < 0): w = w + η × |R| × (1 - w)
        
        These rules have the property that weight changes naturally decelerate
        as weights approach boundaries (0 or 1).
        
        Only weights for active KCs are modified.
        
        Args:
            kc_active: Binary KC activation pattern of shape (n_kc,)
            modulatory_signal: Modulatory signal R (positive = LTD, negative = LTP)
        """
        # Create mask for active KCs (shape: n_kc, 1 for broadcasting)
        active_mask = kc_active.reshape(-1, 1) > 0
        
        if modulatory_signal > 0:  # Aversive (LTD)
            # Multiplicative LTD rule: w = w × (1 - η × R)
            # Weight decreases proportionally to current weight
            decay_factor = 1 - self.learning_rate * modulatory_signal
            self._W_kc_mbon = np.where(
                active_mask,
                self._W_kc_mbon * decay_factor,
                self._W_kc_mbon
            )
        else:  # Appetitive (LTP) - modulatory_signal <= 0
            # Multiplicative LTP rule: w = w + η × |R| × (1 - w)
            # Weight increases proportionally to distance from 1
            growth = self.learning_rate * abs(modulatory_signal) * (1 - self._W_kc_mbon)
            self._W_kc_mbon = np.where(
                active_mask,
                self._W_kc_mbon + growth,
                self._W_kc_mbon
            )
        
        # Clip weights to [0, 1] range (safety measure)
        self._W_kc_mbon = np.clip(self._W_kc_mbon, 0.0, 1.0)
    
    def train_aversive(self, odor: np.ndarray, strength: float = 1.0) -> float:
        """
        Execute aversive learning training (LTD).
        
        Pairs the odor with a positive modulatory signal (aversive stimulus),
        which decreases weights for active KCs.
        
        Args:
            odor: Odor input vector of shape (n_pn,)
            strength: Strength of the aversive stimulus (default 1.0)
            
        Returns:
            Total weight change magnitude
        """
        # Get KC activation for this odor
        _, kc_activation = self.predict(odor)
        
        # Store weights before update
        weights_before = self._W_kc_mbon.copy()
        
        # Apply positive modulatory signal (aversive -> LTD)
        self.modulate(kc_activation, modulatory_signal=strength)
        
        # Calculate total weight change
        weight_change = np.sum(np.abs(self._W_kc_mbon - weights_before))
        
        # Record learning event (Requirements 5.2)
        self._learning_history.append({
            'type': 'aversive',
            'odor_hash': hash(odor.tobytes()),
            'strength': strength,
            'weight_change': float(weight_change),
            'timestamp': time.time()
        })
        
        return weight_change
    
    def train_appetitive(self, odor: np.ndarray, strength: float = 1.0) -> float:
        """
        Execute appetitive learning training (LTP).
        
        Pairs the odor with a negative modulatory signal (appetitive stimulus),
        which increases weights for active KCs.
        
        Args:
            odor: Odor input vector of shape (n_pn,)
            strength: Strength of the appetitive stimulus (default 1.0)
            
        Returns:
            Total weight change magnitude
        """
        # Get KC activation for this odor
        _, kc_activation = self.predict(odor)
        
        # Store weights before update
        weights_before = self._W_kc_mbon.copy()
        
        # Apply negative modulatory signal (appetitive -> LTP)
        self.modulate(kc_activation, modulatory_signal=-strength)
        
        # Calculate total weight change
        weight_change = np.sum(np.abs(self._W_kc_mbon - weights_before))
        
        # Record learning event (Requirements 5.2)
        self._learning_history.append({
            'type': 'appetitive',
            'odor_hash': hash(odor.tobytes()),
            'strength': strength,
            'weight_change': float(weight_change),
            'timestamp': time.time()
        })
        
        return weight_change
    
    def reset_weights(self, clear_history: bool = False) -> None:
        """
        Reset KC-MBON weights to initial state (all 1.0).
        
        Args:
            clear_history: If True, also clear the learning history (Requirements 5.4)
        """
        self._W_kc_mbon = self._W_kc_mbon_initial.copy()
        if clear_history:
            self._learning_history = []
    
    def get_learning_history(self) -> List[LearningEvent]:
        """
        返回学习历史记录的副本 (Requirements 5.3)
        
        Returns:
            包含所有训练事件的列表副本
        """
        return self._learning_history.copy()
    
    def to_json(self) -> str:
        """
        Serialize model state to JSON format.
        
        Includes all model parameters, weight matrices (with float64 precision),
        seed value, and learning history for complete state restoration.
        Also includes a 'config' section with all configuration parameters.
        
        Returns:
            JSON string representation of the model state
            
        Requirements: 6.1, 6.2, 7.4
        """
        # Ensure weights are float64 for precision (Requirements 6.2)
        W_pn_kc = self._encoder.weights.astype(np.float64)
        W_kc_mbon = self._W_kc_mbon.astype(np.float64)
        
        data = {
            "n_pn": self.n_pn,
            "n_kc": self.n_kc,
            "n_mbon": self.n_mbon,
            "sparsity": self.sparsity,
            "learning_rate": self.learning_rate,
            "connectivity": self.connectivity,
            "seed": self._seed,  # Requirements 6.1: Include seed for complete restoration
            "W_pn_kc": W_pn_kc.tolist(),
            "W_kc_mbon": W_kc_mbon.tolist(),
            "learning_history": self._learning_history,  # Include learning history
            # Include complete configuration for easy restoration (Requirements 7.4)
            "config": {
                "n_pn": self.n_pn,
                "n_kc": self.n_kc,
                "n_mbon": self.n_mbon,
                "sparsity": self.sparsity,
                "learning_rate": self.learning_rate,
                "connectivity": self.connectivity,
                "seed": self._seed
            }
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DrosophilaOlfactoryModel':
        """
        Restore model from JSON format.
        
        Args:
            json_str: JSON string representation of the model state
            
        Returns:
            DrosophilaOlfactoryModel instance with restored state
            
        Raises:
            KeyError: If required fields are missing (Requirements 6.3)
            ValueError: If weight matrix dimensions don't match parameters (Requirements 1.1, 1.2, 6.4)
            TypeError: If field types don't match expected types
        """
        data = json.loads(json_str)
        
        # Validate required fields (Requirements 6.3)
        required_fields = ["n_pn", "n_kc", "n_mbon", "sparsity", "learning_rate", "W_pn_kc", "W_kc_mbon"]
        for field in required_fields:
            if field not in data:
                raise KeyError(f"Missing required field: {field}")
        
        # Convert weight matrices to numpy arrays with float64 precision
        W_pn_kc = np.array(data["W_pn_kc"], dtype=np.float64)
        W_kc_mbon = np.array(data["W_kc_mbon"], dtype=np.float64)
        
        # Validate weight matrix dimensions (Requirements 1.1, 1.2, 6.4)
        expected_pn_kc_shape = (data["n_pn"], data["n_kc"])
        if W_pn_kc.shape != expected_pn_kc_shape:
            raise ValueError(
                f"W_pn_kc shape {W_pn_kc.shape} doesn't match "
                f"expected {expected_pn_kc_shape}"
            )
        
        expected_kc_mbon_shape = (data["n_kc"], data["n_mbon"])
        if W_kc_mbon.shape != expected_kc_mbon_shape:
            raise ValueError(
                f"W_kc_mbon shape {W_kc_mbon.shape} doesn't match "
                f"expected {expected_kc_mbon_shape}"
            )
        
        # Create model with parameters (including seed if available)
        model = cls(
            n_pn=data["n_pn"],
            n_kc=data["n_kc"],
            n_mbon=data["n_mbon"],
            sparsity=data["sparsity"],
            learning_rate=data["learning_rate"],
            connectivity=data.get("connectivity", 0.14),
            seed=data.get("seed", None)
        )
        
        # Restore weight matrices
        model._encoder._weights = W_pn_kc
        model._W_kc_mbon = W_kc_mbon
        
        # Correctly set W_kc_mbon_initial to a copy of restored weights (Requirements 1.4)
        model._W_kc_mbon_initial = W_kc_mbon.copy()
        
        # Restore learning history if available
        model._learning_history = data.get("learning_history", [])
        
        return model
