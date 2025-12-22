# Shared fixtures and generators for property-based testing
"""
Custom Hypothesis strategies for generating test data.
"""

import numpy as np
from hypothesis import strategies as st


@st.composite
def odor_vectors(draw, n_features=50):
    """Generate valid odor vectors with values in [0, 1]"""
    values = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=n_features, max_size=n_features
    ))
    return np.array(values)


@st.composite
def similar_odor_pairs(draw, n_features=50, min_overlap=0.8):
    """Generate pairs of similar odors with controlled overlap"""
    base = draw(odor_vectors(n_features))
    n_changed = int(n_features * (1 - min_overlap))
    variant = base.copy()
    indices = draw(st.lists(
        st.integers(0, n_features - 1),
        min_size=n_changed, max_size=n_changed, unique=True
    ))
    for idx in indices:
        variant[idx] = draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    return base, np.clip(variant, 0, 1)


@st.composite
def model_configs(draw):
    """Generate valid model configurations"""
    return {
        'n_pn': draw(st.integers(10, 100)),
        'n_kc': draw(st.integers(100, 500)),
        'n_mbon': draw(st.integers(1, 5)),
        'sparsity': draw(st.floats(0.01, 0.2, allow_nan=False, allow_infinity=False)),
        'learning_rate': draw(st.floats(0.001, 0.5, allow_nan=False, allow_infinity=False))
    }
