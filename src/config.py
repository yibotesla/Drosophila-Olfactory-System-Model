"""
ModelConfig - Configuration dataclass for the Drosophila Olfactory Model.

Provides centralized configuration management with validation for all model parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration parameters for DrosophilaOlfactoryModel.
    
    Attributes:
        n_pn: Number of projection neurons (input dimension). Default 50.
        n_kc: Number of Kenyon cells. Default 2000.
        n_mbon: Number of mushroom body output neurons. Default 1.
        sparsity: Fraction of KCs to keep active (kWTA). Default 0.05 (5%).
        learning_rate: Learning rate (η) for weight updates. Default 0.05.
        connectivity: Fraction of PNs each KC connects to. Default 0.14.
        seed: Optional random seed for reproducibility.
    """
    n_pn: int = 50
    n_kc: int = 2000
    n_mbon: int = 1
    sparsity: float = 0.05
    learning_rate: float = 0.05
    connectivity: float = 0.14
    seed: Optional[int] = None
    
    def validate(self) -> None:
        """
        Validate all configuration parameters.
        
        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if self.n_pn <= 0:
            raise ValueError(f"n_pn must be positive, got {self.n_pn}")
        if self.n_kc <= 0:
            raise ValueError(f"n_kc must be positive, got {self.n_kc}")
        if self.n_mbon <= 0:
            raise ValueError(f"n_mbon must be positive, got {self.n_mbon}")
        if not (0 < self.sparsity < 1):
            raise ValueError(f"sparsity must be in (0, 1), got {self.sparsity}")
        if self.learning_rate < 0:
            raise ValueError(f"learning_rate must be non-negative, got {self.learning_rate}")
        if not (0 < self.connectivity <= 1):
            raise ValueError(f"connectivity must be in (0, 1], got {self.connectivity}")
