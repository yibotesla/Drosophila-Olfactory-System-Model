# Drosophila Olfactory Model Package
"""
A computational model of the Drosophila olfactory system.

This package implements a three-layer feedforward network simulating:
- Antennal Lobe (AL) -> Mushroom Body (MB) -> Output Neurons (MBONs)
- Dopaminergic neuron (DAN) modulation for associative learning
"""

__version__ = "0.1.0"

from src.odor_dataset import OdorDataset
from src.model import DrosophilaOlfactoryModel
from src.sparse_encoder import SparseEncoder
from src.evaluator import ModelEvaluator

__all__ = ["OdorDataset", "DrosophilaOlfactoryModel", "SparseEncoder", "ModelEvaluator"]
