"""
ModelEvaluator class for evaluating the Drosophila Olfactory Model.

This module provides evaluation methods for:
- Computing discrimination index (learning effect)
- Measuring pattern separation in KC layer
- Evaluating generalization to odor variants
- Evaluating learning specificity
"""

from typing import Dict, List
import numpy as np

from src.model import DrosophilaOlfactoryModel


class ModelEvaluator:
    """
    模型评估器 - Model evaluator for the Drosophila Olfactory Model.
    
    Provides methods to evaluate model performance including:
    - Discrimination index for learning effect
    - Pattern separation measurement
    - Generalization ability
    - Learning specificity
    """
    
    def __init__(self, model: DrosophilaOlfactoryModel):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model: DrosophilaOlfactoryModel instance to evaluate
        """
        if model is None:
            raise ValueError("model cannot be None")
        self.model = model
    
    def compute_discrimination_index(
        self, 
        response_before: np.ndarray,
        response_after: np.ndarray,
        mbon_idx: int = 0
    ) -> float:
        """
        Compute discrimination index measuring learning effect for a specific MBON.
        
        The discrimination index quantifies the change in MBON output
        before and after training. A positive value indicates reduced
        response (aversive learning), negative indicates increased response.
        
        Formula: DI = (response_before - response_after) / response_before
        
        Args:
            response_before: MBON output before training, shape (n_mbon,) or scalar
            response_after: MBON output after training, shape (n_mbon,) or scalar
            mbon_idx: Index of the MBON to compute discrimination index for (default 0)
            
        Returns:
            Discrimination index in range [-inf, 1], where:
            - Positive values indicate response reduction (aversive learning)
            - Negative values indicate response increase (appetitive learning)
            - 0 indicates no change
            
        Raises:
            ValueError: If response_before is zero (division by zero)
            ValueError: If mbon_idx is out of range for the response arrays
        """
        # Handle scalar inputs for backward compatibility
        if np.isscalar(response_before):
            response_before = np.array([response_before])
        if np.isscalar(response_after):
            response_after = np.array([response_after])
        
        # Ensure numpy arrays
        response_before = np.atleast_1d(response_before)
        response_after = np.atleast_1d(response_after)
        
        # Validate mbon_idx (Requirements 8.3)
        if mbon_idx < 0 or mbon_idx >= len(response_before):
            raise ValueError(
                f"mbon_idx {mbon_idx} out of range for {len(response_before)} MBONs"
            )
        
        if mbon_idx >= len(response_after):
            raise ValueError(
                f"mbon_idx {mbon_idx} out of range for {len(response_after)} MBONs"
            )
        
        before = response_before[mbon_idx]
        after = response_after[mbon_idx]
        
        if before == 0:
            raise ValueError("response_before cannot be zero")
        
        return (before - after) / before

    
    def compute_pattern_separation(
        self, 
        odor_a: np.ndarray, 
        odor_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute pattern separation effect in the KC layer.
        
        Measures how the KC layer transforms similar inputs into more
        distinct representations. Pattern separation is a key function
        of the mushroom body.
        
        Args:
            odor_a: First odor vector of shape (n_pn,)
            odor_b: Second odor vector of shape (n_pn,)
            
        Returns:
            Dictionary containing:
            - 'input_distance': Euclidean distance between input odors
            - 'kc_distance': Euclidean distance between KC representations
            - 'separation_ratio': kc_distance / input_distance (>1 means separation)
            - 'input_overlap': Cosine similarity of inputs
            - 'kc_overlap': Cosine similarity of KC patterns
            
        Raises:
            ValueError: If odor dimensions don't match model's n_pn
        """
        if odor_a.shape[0] != self.model.n_pn:
            raise ValueError(
                f"odor_a dimension {odor_a.shape[0]} doesn't match n_pn {self.model.n_pn}"
            )
        if odor_b.shape[0] != self.model.n_pn:
            raise ValueError(
                f"odor_b dimension {odor_b.shape[0]} doesn't match n_pn {self.model.n_pn}"
            )
        
        # Get KC activations for both odors
        _, kc_a = self.model.predict(odor_a)
        _, kc_b = self.model.predict(odor_b)
        
        # Compute Euclidean distances
        input_distance = np.linalg.norm(odor_a - odor_b)
        kc_distance = np.linalg.norm(kc_a - kc_b)
        
        # Compute separation ratio (avoid division by zero)
        if input_distance > 0:
            separation_ratio = kc_distance / input_distance
        else:
            separation_ratio = 0.0 if kc_distance == 0 else float('inf')
        
        # Compute cosine similarities (overlap)
        def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))
        
        input_overlap = cosine_similarity(odor_a, odor_b)
        kc_overlap = cosine_similarity(kc_a, kc_b)
        
        return {
            'input_distance': float(input_distance),
            'kc_distance': float(kc_distance),
            'separation_ratio': float(separation_ratio),
            'input_overlap': float(input_overlap),
            'kc_overlap': float(kc_overlap)
        }
    
    def evaluate_generalization(
        self, 
        trained_odor: np.ndarray, 
        test_variants: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate model's generalization to variants of a trained odor.
        
        Tests how the model responds to noisy variants of an odor
        after training on the original. Good generalization means
        similar responses to similar odors.
        
        Args:
            trained_odor: The odor used for training, shape (n_pn,)
            test_variants: Array of variant odors, shape (n_variants, n_pn)
            
        Returns:
            np.ndarray: MBON responses to each variant
                - For single MBON (n_mbon=1): shape (n_variants,) for backward compatibility
                - For multiple MBONs (n_mbon>1): shape (n_variants, n_mbon)
            
        Raises:
            ValueError: If dimensions don't match model's n_pn
        """
        if trained_odor.shape[0] != self.model.n_pn:
            raise ValueError(
                f"trained_odor dimension {trained_odor.shape[0]} doesn't match n_pn {self.model.n_pn}"
            )
        
        if test_variants.ndim == 1:
            test_variants = test_variants.reshape(1, -1)
        
        if test_variants.shape[1] != self.model.n_pn:
            raise ValueError(
                f"test_variants dimension {test_variants.shape[1]} doesn't match n_pn {self.model.n_pn}"
            )
        
        responses = []
        for variant in test_variants:
            output, _ = self.model.predict(variant)
            responses.append(output)  # Keep all MBON outputs (Requirements 8.4)
        
        responses_array = np.array(responses)
        
        # For backward compatibility: if n_mbon=1, return shape (n_variants,)
        if self.model.n_mbon == 1:
            return responses_array.flatten()
        
        # For multi-MBON: return shape (n_variants, n_mbon)
        return responses_array
    
    def evaluate_specificity(
        self, 
        trained_odor: np.ndarray, 
        untrained_odors: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate learning specificity by comparing trained vs untrained odor responses.
        
        Measures whether learning is specific to the trained odor or
        generalizes inappropriately to unrelated odors.
        
        Args:
            trained_odor: The odor used for training, shape (n_pn,)
            untrained_odors: List of odors NOT used in training
            
        Returns:
            Dictionary containing:
            - 'trained_response': MBON response to trained odor
            - 'untrained_mean': Mean MBON response to untrained odors
            - 'untrained_std': Std dev of responses to untrained odors
            - 'specificity_index': Difference between untrained and trained responses
              (higher = more specific learning)
            
        Raises:
            ValueError: If dimensions don't match or untrained_odors is empty
        """
        if trained_odor.shape[0] != self.model.n_pn:
            raise ValueError(
                f"trained_odor dimension {trained_odor.shape[0]} doesn't match n_pn {self.model.n_pn}"
            )
        
        if not untrained_odors:
            raise ValueError("untrained_odors cannot be empty")
        
        # Get response to trained odor
        trained_output, _ = self.model.predict(trained_odor)
        trained_response = float(trained_output[0])
        
        # Get responses to untrained odors
        untrained_responses = []
        for odor in untrained_odors:
            if odor.shape[0] != self.model.n_pn:
                raise ValueError(
                    f"untrained odor dimension {odor.shape[0]} doesn't match n_pn {self.model.n_pn}"
                )
            output, _ = self.model.predict(odor)
            untrained_responses.append(output[0])
        
        untrained_responses = np.array(untrained_responses)
        untrained_mean = float(np.mean(untrained_responses))
        untrained_std = float(np.std(untrained_responses))
        
        # Specificity index: how much higher untrained responses are vs trained
        # Higher value = more specific learning (trained odor has lower response)
        specificity_index = untrained_mean - trained_response
        
        return {
            'trained_response': trained_response,
            'untrained_mean': untrained_mean,
            'untrained_std': untrained_std,
            'specificity_index': float(specificity_index)
        }
