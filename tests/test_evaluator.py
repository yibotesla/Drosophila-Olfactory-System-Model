"""
Unit tests for ModelEvaluator class.

Tests cover:
- Discrimination index calculation (Requirements 6.2)
- Pattern separation measurement (Requirements 6.1)
- Generalization evaluation (Requirements 6.3)
- Specificity evaluation (Requirements 6.4)
"""

import numpy as np
import pytest

from src.model import DrosophilaOlfactoryModel
from src.evaluator import ModelEvaluator


class TestDiscriminationIndex:
    """Tests for compute_discrimination_index method."""
    
    def test_positive_discrimination_index(self):
        """Test that reduced response gives positive discrimination index."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        # Response decreased from 100 to 60 (40% reduction)
        di = evaluator.compute_discrimination_index(100.0, 60.0)
        assert di == pytest.approx(0.4)
    
    def test_negative_discrimination_index(self):
        """Test that increased response gives negative discrimination index."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        # Response increased from 100 to 150 (50% increase)
        di = evaluator.compute_discrimination_index(100.0, 150.0)
        assert di == pytest.approx(-0.5)
    
    def test_zero_change_discrimination_index(self):
        """Test that no change gives zero discrimination index."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        di = evaluator.compute_discrimination_index(100.0, 100.0)
        assert di == pytest.approx(0.0)
    
    def test_complete_suppression(self):
        """Test discrimination index when response is completely suppressed."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        # Response reduced to zero (100% reduction)
        di = evaluator.compute_discrimination_index(100.0, 0.0)
        assert di == pytest.approx(1.0)
    
    def test_zero_before_raises_error(self):
        """Test that zero response_before raises ValueError."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        with pytest.raises(ValueError, match="response_before cannot be zero"):
            evaluator.compute_discrimination_index(0.0, 50.0)


class TestPatternSeparation:
    """Tests for compute_pattern_separation method."""
    
    def test_pattern_separation_returns_all_keys(self):
        """Test that pattern separation returns all expected metrics."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        odor_a = np.random.RandomState(1).uniform(0, 1, 50)
        odor_b = np.random.RandomState(2).uniform(0, 1, 50)
        
        result = evaluator.compute_pattern_separation(odor_a, odor_b)
        
        expected_keys = ['input_distance', 'kc_distance', 'separation_ratio', 
                        'input_overlap', 'kc_overlap']
        assert all(key in result for key in expected_keys)
    
    def test_identical_odors_zero_distance(self):
        """Test that identical odors have zero distance."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        odor = np.random.RandomState(1).uniform(0, 1, 50)
        
        result = evaluator.compute_pattern_separation(odor, odor)
        
        assert result['input_distance'] == pytest.approx(0.0)
        assert result['kc_distance'] == pytest.approx(0.0)
        assert result['input_overlap'] == pytest.approx(1.0)
        assert result['kc_overlap'] == pytest.approx(1.0)
    
    def test_pattern_separation_effect(self):
        """Test that KC layer provides pattern separation for similar odors."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=500, sparsity=0.05, seed=42)
        evaluator = ModelEvaluator(model)
        
        # Create similar odors (80% overlap)
        rng = np.random.RandomState(1)
        odor_a = rng.uniform(0, 1, 50)
        odor_b = odor_a.copy()
        # Change 20% of values
        indices = rng.choice(50, 10, replace=False)
        odor_b[indices] = rng.uniform(0, 1, 10)
        
        result = evaluator.compute_pattern_separation(odor_a, odor_b)
        
        # KC distance should be greater than input distance (pattern separation)
        # This is the key property of the mushroom body
        assert result['separation_ratio'] > 1.0
    
    def test_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises ValueError."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        odor_a = np.random.uniform(0, 1, 50)
        odor_b = np.random.uniform(0, 1, 30)  # Wrong dimension
        
        with pytest.raises(ValueError, match="doesn't match n_pn"):
            evaluator.compute_pattern_separation(odor_a, odor_b)



class TestEvaluateGeneralization:
    """Tests for evaluate_generalization method."""
    
    def test_generalization_returns_correct_shape(self):
        """Test that generalization returns array of correct shape."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        trained_odor = np.random.RandomState(1).uniform(0, 1, 50)
        # Create 5 variants
        variants = np.random.RandomState(2).uniform(0, 1, (5, 50))
        
        responses = evaluator.evaluate_generalization(trained_odor, variants)
        
        assert responses.shape == (5,)
    
    def test_generalization_single_variant(self):
        """Test generalization with a single variant."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        trained_odor = np.random.RandomState(1).uniform(0, 1, 50)
        variant = np.random.RandomState(2).uniform(0, 1, 50)
        
        # Single variant as 1D array
        responses = evaluator.evaluate_generalization(trained_odor, variant)
        
        assert responses.shape == (1,)
    
    def test_generalization_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        trained_odor = np.random.uniform(0, 1, 50)
        variants = np.random.uniform(0, 1, (5, 30))  # Wrong dimension
        
        with pytest.raises(ValueError, match="doesn't match n_pn"):
            evaluator.evaluate_generalization(trained_odor, variants)


class TestEvaluateSpecificity:
    """Tests for evaluate_specificity method."""
    
    def test_specificity_returns_all_keys(self):
        """Test that specificity returns all expected metrics."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        trained_odor = np.random.RandomState(1).uniform(0, 1, 50)
        untrained_odors = [
            np.random.RandomState(i).uniform(0, 1, 50) 
            for i in range(2, 5)
        ]
        
        result = evaluator.evaluate_specificity(trained_odor, untrained_odors)
        
        expected_keys = ['trained_response', 'untrained_mean', 
                        'untrained_std', 'specificity_index']
        assert all(key in result for key in expected_keys)
    
    def test_specificity_after_aversive_training(self):
        """Test specificity after aversive training shows positive index."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, learning_rate=0.1, seed=42)
        evaluator = ModelEvaluator(model)
        
        # Create distinct odors
        rng = np.random.RandomState(1)
        trained_odor = rng.uniform(0, 1, 50)
        untrained_odors = [rng.uniform(0, 1, 50) for _ in range(3)]
        
        # Train on one odor (aversive)
        model.train_aversive(trained_odor, strength=1.0)
        
        result = evaluator.evaluate_specificity(trained_odor, untrained_odors)
        
        # After aversive training, trained response should be lower
        # So specificity_index (untrained_mean - trained_response) should be positive
        assert result['specificity_index'] > 0
    
    def test_specificity_empty_untrained_raises_error(self):
        """Test that empty untrained_odors raises ValueError."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        trained_odor = np.random.uniform(0, 1, 50)
        
        with pytest.raises(ValueError, match="untrained_odors cannot be empty"):
            evaluator.evaluate_specificity(trained_odor, [])
    
    def test_specificity_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        trained_odor = np.random.uniform(0, 1, 50)
        untrained_odors = [np.random.uniform(0, 1, 30)]  # Wrong dimension
        
        with pytest.raises(ValueError, match="doesn't match n_pn"):
            evaluator.evaluate_specificity(trained_odor, untrained_odors)


class TestModelEvaluatorInit:
    """Tests for ModelEvaluator initialization."""
    
    def test_init_with_valid_model(self):
        """Test initialization with valid model."""
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=200, seed=42)
        evaluator = ModelEvaluator(model)
        
        assert evaluator.model is model
    
    def test_init_with_none_raises_error(self):
        """Test that None model raises ValueError."""
        with pytest.raises(ValueError, match="model cannot be None"):
            ModelEvaluator(None)
