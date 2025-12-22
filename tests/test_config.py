"""
Property-based tests for ModelConfig.

Uses Hypothesis for property-based testing with minimum 100 iterations per property.
"""

import pytest
from hypothesis import given, strategies as st, settings

from src.config import ModelConfig


# =============================================================================
# Property 11: Config Validation
# =============================================================================

# **Feature: model-improvements, Property 11: Config Validation**
# **Validates: Requirements 7.2, 7.3**


@st.composite
def invalid_n_pn_configs(draw):
    """Generate configs with invalid n_pn (non-positive)."""
    config = ModelConfig()
    config.n_pn = draw(st.integers(max_value=0))
    return config


@st.composite
def invalid_n_kc_configs(draw):
    """Generate configs with invalid n_kc (non-positive)."""
    config = ModelConfig()
    config.n_kc = draw(st.integers(max_value=0))
    return config


@st.composite
def invalid_n_mbon_configs(draw):
    """Generate configs with invalid n_mbon (non-positive)."""
    config = ModelConfig()
    config.n_mbon = draw(st.integers(max_value=0))
    return config


@st.composite
def invalid_sparsity_configs(draw):
    """Generate configs with invalid sparsity (not in (0, 1))."""
    config = ModelConfig()
    # Either <= 0 or >= 1
    config.sparsity = draw(st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    return config


@st.composite
def invalid_learning_rate_configs(draw):
    """Generate configs with invalid learning_rate (negative)."""
    config = ModelConfig()
    config.learning_rate = draw(st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
    return config


@st.composite
def invalid_connectivity_configs(draw):
    """Generate configs with invalid connectivity (not in (0, 1])."""
    config = ModelConfig()
    # Either <= 0 or > 1
    config.connectivity = draw(st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.001, allow_nan=False, allow_infinity=False)
    ))
    return config


@settings(max_examples=100)
@given(config=invalid_n_pn_configs())
def test_property_11_invalid_n_pn_raises_error(config: ModelConfig):
    """
    Property 11: Config Validation - Invalid n_pn
    
    For any ModelConfig with n_pn <= 0, validate() SHALL raise ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "n_pn" in str(exc_info.value)


@settings(max_examples=100)
@given(config=invalid_n_kc_configs())
def test_property_11_invalid_n_kc_raises_error(config: ModelConfig):
    """
    Property 11: Config Validation - Invalid n_kc
    
    For any ModelConfig with n_kc <= 0, validate() SHALL raise ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "n_kc" in str(exc_info.value)


@settings(max_examples=100)
@given(config=invalid_n_mbon_configs())
def test_property_11_invalid_n_mbon_raises_error(config: ModelConfig):
    """
    Property 11: Config Validation - Invalid n_mbon
    
    For any ModelConfig with n_mbon <= 0, validate() SHALL raise ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "n_mbon" in str(exc_info.value)


@settings(max_examples=100)
@given(config=invalid_sparsity_configs())
def test_property_11_invalid_sparsity_raises_error(config: ModelConfig):
    """
    Property 11: Config Validation - Invalid sparsity
    
    For any ModelConfig with sparsity not in (0, 1), validate() SHALL raise ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "sparsity" in str(exc_info.value)


@settings(max_examples=100)
@given(config=invalid_learning_rate_configs())
def test_property_11_invalid_learning_rate_raises_error(config: ModelConfig):
    """
    Property 11: Config Validation - Invalid learning_rate
    
    For any ModelConfig with learning_rate < 0, validate() SHALL raise ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "learning_rate" in str(exc_info.value)


@settings(max_examples=100)
@given(config=invalid_connectivity_configs())
def test_property_11_invalid_connectivity_raises_error(config: ModelConfig):
    """
    Property 11: Config Validation - Invalid connectivity
    
    For any ModelConfig with connectivity not in (0, 1], validate() SHALL raise ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "connectivity" in str(exc_info.value)


@st.composite
def valid_model_configs(draw):
    """Generate valid ModelConfig instances."""
    return ModelConfig(
        n_pn=draw(st.integers(min_value=1, max_value=200)),
        n_kc=draw(st.integers(min_value=1, max_value=5000)),
        n_mbon=draw(st.integers(min_value=1, max_value=10)),
        sparsity=draw(st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False)),
        learning_rate=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        connectivity=draw(st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False)),
        seed=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)))
    )


@settings(max_examples=100)
@given(config=valid_model_configs())
def test_property_11_valid_config_passes_validation(config: ModelConfig):
    """
    Property 11: Config Validation - Valid configs
    
    For any ModelConfig with all valid parameters, validate() SHALL NOT raise any exception.
    """
    # Should not raise any exception
    config.validate()


# =============================================================================
# Property 12: Config Application
# =============================================================================

# **Feature: model-improvements, Property 12: Config Application**
# **Validates: Requirements 7.4**

from src.model import DrosophilaOlfactoryModel


@settings(max_examples=100)
@given(config=valid_model_configs())
def test_property_12_config_application(config: ModelConfig):
    """
    Property 12: Config Application
    
    For any valid ModelConfig, a model created from it SHALL have all parameters
    matching the config values.
    """
    # Validate config first
    config.validate()
    
    # Create model using from_config class method (Requirements 7.4)
    model = DrosophilaOlfactoryModel.from_config(config)
    
    # Verify all parameters match
    assert model.n_pn == config.n_pn, \
        f"n_pn mismatch: model={model.n_pn}, config={config.n_pn}"
    assert model.n_kc == config.n_kc, \
        f"n_kc mismatch: model={model.n_kc}, config={config.n_kc}"
    assert model.n_mbon == config.n_mbon, \
        f"n_mbon mismatch: model={model.n_mbon}, config={config.n_mbon}"
    assert model.sparsity == config.sparsity, \
        f"sparsity mismatch: model={model.sparsity}, config={config.sparsity}"
    assert model.learning_rate == config.learning_rate, \
        f"learning_rate mismatch: model={model.learning_rate}, config={config.learning_rate}"
    assert model.connectivity == config.connectivity, \
        f"connectivity mismatch: model={model.connectivity}, config={config.connectivity}"
