"""
SparseEncoder class implementing the KC (Kenyon Cell) layer.

This module provides sparse encoding functionality using k-Winner-Take-All (kWTA)
mechanism to simulate the sparse activation patterns in the Drosophila mushroom body.
"""

import numpy as np
from numpy.random import Generator, PCG64
from typing import Optional


class SparseEncoder:
    """
    稀疏编码器 (KC层) - Sparse encoder implementing the Kenyon Cell layer.
    
    Uses k-Winner-Take-All (kWTA) mechanism to produce sparse activation patterns,
    where only the top k neurons (approximately 5% by default) are active.
    """
    
    def __init__(
        self, 
        n_input: int, 
        n_output: int, 
        sparsity: float = 0.05,
        connectivity: float = 0.14,
        seed: Optional[int] = None
    ):
        """
        Initialize the SparseEncoder.
        
        Args:
            n_input: Number of input neurons (PNs)
            n_output: Number of output neurons (KCs)
            sparsity: Fraction of KCs to keep active (default 0.05 = 5%)
            connectivity: Fraction of PNs each KC connects to (default 0.14 ≈ 7/50)
            seed: Optional random seed for reproducibility
            
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        if n_input <= 0:
            raise ValueError(f"n_input must be positive, got {n_input}")
        if n_output <= 0:
            raise ValueError(f"n_output must be positive, got {n_output}")
        if not (0 < sparsity < 1):
            raise ValueError(f"sparsity must be in (0, 1), got {sparsity}")
        if not (0 < connectivity <= 1):
            raise ValueError(f"connectivity must be in (0, 1], got {connectivity}")
        
        self.n_input = n_input
        self.n_output = n_output
        self.sparsity = sparsity
        self.connectivity = connectivity
        self._seed = seed
        
        # Calculate k for kWTA
        self._k = max(1, int(np.floor(n_output * sparsity)))
        
        # Initialize independent random number generator (Requirements 2.1, 2.2, 2.3)
        if seed is not None:
            self._rng = Generator(PCG64(seed))
        else:
            self._rng = np.random.default_rng()
        
        # Initialize weight matrix using the independent RNG
        self._weights = self._initialize_weights()
    
    def _initialize_weights(self) -> np.ndarray:
        """
        Initialize sparse random connection matrix W_PN_KC.
        
        Each KC connects to approximately (connectivity * n_input) PNs randomly.
        The connections are binary (0 or 1).
        
        Uses the instance's independent RNG to avoid polluting global random state.
            
        Returns:
            np.ndarray: Binary weight matrix of shape (n_input, n_output)
        """
        # Create sparse random connections
        # Each column (KC) has approximately connectivity * n_input connections
        weights = np.zeros((self.n_input, self.n_output), dtype=np.float64)
        
        for kc_idx in range(self.n_output):
            # Determine number of connections for this KC
            n_connections = max(1, int(np.round(self.connectivity * self.n_input)))
            # Randomly select which PNs to connect to using instance RNG
            connected_pns = self._rng.choice(
                self.n_input, 
                size=n_connections, 
                replace=False
            )
            weights[connected_pns, kc_idx] = 1.0
        
        return weights
    
    def encode(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Encode input using k-Winner-Take-All mechanism.
        
        Projects the input through the sparse connection matrix and applies
        kWTA to keep only the top k activations.
        
        Args:
            input_vector: Input odor vector of shape (n_input,)
            
        Returns:
            np.ndarray: Sparse KC activation of shape (n_output,) with exactly k 
                       active neurons (value 1.0) and rest inactive (value 0.0)
                       
        Raises:
            ValueError: If input dimension doesn't match n_input
        """
        if input_vector.shape[0] != self.n_input:
            raise ValueError(
                f"Input dimension {input_vector.shape[0]} doesn't match n_input {self.n_input}"
            )
        
        # Project input through weight matrix: (n_input,) @ (n_input, n_output) -> (n_output,)
        projection = input_vector @ self._weights
        
        # Apply k-Winner-Take-All
        # Find indices of top k activations
        if self._k >= self.n_output:
            # All neurons active (edge case)
            return np.ones(self.n_output, dtype=np.float64)
        
        # Get indices of k largest values
        top_k_indices = np.argpartition(projection, -self._k)[-self._k:]
        
        # Create sparse output
        output = np.zeros(self.n_output, dtype=np.float64)
        output[top_k_indices] = 1.0
        
        return output
    
    def get_active_indices(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Get indices of active KCs for a given input.
        
        Args:
            input_vector: Input odor vector of shape (n_input,)
            
        Returns:
            np.ndarray: Array of indices of active KCs (length k)
        """
        activation = self.encode(input_vector)
        return np.where(activation > 0)[0]
    
    @property
    def weights(self) -> np.ndarray:
        """Get the connection weight matrix (read-only copy)."""
        return self._weights.copy()
    
    @property
    def k(self) -> int:
        """Get the number of active neurons in kWTA."""
        return self._k
