"""
OdorDataset class for generating and managing odor data.

This module provides functionality for:
- Generating odor prototype vectors
- Creating variants with Gaussian noise
- Creating concentration variants with scaling
- Dataset serialization/deserialization
"""

import json
from typing import Dict, List, Tuple, Optional
import numpy as np


class OdorDataset:
    """气味数据集生成器 - Odor dataset generator"""
    
    def __init__(self, n_features: int = 50):
        """
        Initialize the OdorDataset.
        
        Args:
            n_features: Number of features (glomeruli) in odor vectors. Default 50.
        """
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        
        self.n_features = n_features
        self._prototypes: Dict[str, np.ndarray] = {}
        self._samples: Optional[np.ndarray] = None
        self._labels: Optional[List[str]] = None
    
    def generate_prototype(self, name: str, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate an odor prototype vector with random values in [0, 1].
        
        Args:
            name: Name identifier for the odor prototype
            seed: Optional random seed for reproducibility
            
        Returns:
            np.ndarray: Odor prototype vector of shape (n_features,) with values in [0, 1]
        """
        if seed is not None:
            np.random.seed(seed)
        
        prototype = np.random.uniform(0.0, 1.0, size=self.n_features)
        self._prototypes[name] = prototype
        return prototype

    def generate_variants(
        self, 
        prototype: np.ndarray, 
        n_samples: int, 
        noise_level: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate variant samples by adding Gaussian noise to a prototype.
        
        Args:
            prototype: Base odor vector to generate variants from
            n_samples: Number of variant samples to generate
            noise_level: Standard deviation of Gaussian noise
            seed: Optional random seed for reproducibility
            
        Returns:
            np.ndarray: Array of shape (n_samples, n_features) with values clipped to [0, 1]
        """
        if prototype.shape[0] != self.n_features:
            raise ValueError(
                f"Prototype dimension {prototype.shape[0]} doesn't match n_features {self.n_features}"
            )
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if noise_level < 0:
            raise ValueError(f"noise_level must be non-negative, got {noise_level}")
        
        if seed is not None:
            np.random.seed(seed)
        
        noise = np.random.normal(0, noise_level, size=(n_samples, self.n_features))
        variants = prototype + noise
        # Clip to ensure values stay in [0, 1]
        return np.clip(variants, 0.0, 1.0)
    
    def generate_concentration_variants(
        self, 
        prototype: np.ndarray, 
        concentration_factors: List[float]
    ) -> np.ndarray:
        """
        Generate concentration variants by scaling the prototype vector.
        
        Args:
            prototype: Base odor vector to scale
            concentration_factors: List of scaling factors for different concentrations
            
        Returns:
            np.ndarray: Array of shape (len(concentration_factors), n_features) 
                       with values clipped to [0, 1]
        """
        if prototype.shape[0] != self.n_features:
            raise ValueError(
                f"Prototype dimension {prototype.shape[0]} doesn't match n_features {self.n_features}"
            )
        if not concentration_factors:
            raise ValueError("concentration_factors cannot be empty")
        
        variants = []
        for factor in concentration_factors:
            if factor < 0:
                raise ValueError(f"Concentration factor must be non-negative, got {factor}")
            scaled = prototype * factor
            variants.append(np.clip(scaled, 0.0, 1.0))
        
        return np.array(variants)

    def create_dataset(
        self, 
        prototypes: Dict[str, np.ndarray],
        n_samples_per_odor: int,
        noise_level: float,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create a complete dataset with multiple odor prototypes and their variants.
        
        Args:
            prototypes: Dictionary mapping odor names to prototype vectors
            n_samples_per_odor: Number of variant samples per odor
            noise_level: Standard deviation of Gaussian noise for variants
            seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of (samples array, labels list)
            - samples: np.ndarray of shape (n_odors * n_samples_per_odor, n_features)
            - labels: List of odor names corresponding to each sample
        """
        if not prototypes:
            raise ValueError("prototypes dictionary cannot be empty")
        
        if seed is not None:
            np.random.seed(seed)
        
        all_samples = []
        all_labels = []
        
        # Store prototypes
        self._prototypes = prototypes.copy()
        
        for name, prototype in prototypes.items():
            variants = self.generate_variants(prototype, n_samples_per_odor, noise_level)
            all_samples.append(variants)
            all_labels.extend([name] * n_samples_per_odor)
        
        self._samples = np.vstack(all_samples)
        self._labels = all_labels
        
        return self._samples, self._labels
    
    def to_json(self) -> str:
        """
        Serialize the dataset to JSON format.
        
        Returns:
            JSON string representation of the dataset
        """
        data = {
            "n_features": self.n_features,
            "prototypes": {
                name: proto.tolist() 
                for name, proto in self._prototypes.items()
            },
            "samples": self._samples.tolist() if self._samples is not None else None,
            "labels": self._labels
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OdorDataset':
        """
        Deserialize a dataset from JSON format.
        
        Args:
            json_str: JSON string representation of the dataset
            
        Returns:
            OdorDataset instance with restored state
        """
        data = json.loads(json_str)
        
        if "n_features" not in data:
            raise KeyError("Missing required field: n_features")
        
        dataset = cls(n_features=data["n_features"])
        
        if "prototypes" in data and data["prototypes"]:
            dataset._prototypes = {
                name: np.array(proto) 
                for name, proto in data["prototypes"].items()
            }
        
        if "samples" in data and data["samples"] is not None:
            dataset._samples = np.array(data["samples"])
        
        if "labels" in data:
            dataset._labels = data["labels"]
        
        return dataset
    
    @property
    def prototypes(self) -> Dict[str, np.ndarray]:
        """Get the stored prototypes."""
        return self._prototypes.copy()
    
    @property
    def samples(self) -> Optional[np.ndarray]:
        """Get the generated samples."""
        return self._samples
    
    @property
    def labels(self) -> Optional[List[str]]:
        """Get the sample labels."""
        return self._labels
