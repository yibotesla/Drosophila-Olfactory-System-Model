# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/`
  - Create `requirements.txt` with numpy, hypothesis dependencies
  - Create `__init__.py` files for package structure
  - _Requirements: All_

- [x] 2. Implement OdorDataset class
  - [x] 2.1 Create OdorDataset with prototype generation
    - Implement `__init__` with n_features parameter
    - Implement `generate_prototype` method returning random vectors in [0, 1]
    - Implement `generate_variants` with Gaussian noise
    - Implement `generate_concentration_variants` with scaling
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [x] 2.2 Write property test for odor vector range
    - **Property 1: Odor Vector Range Invariant**
    - **Validates: Requirements 1.2**
  - [x] 2.3 Implement dataset creation and serialization
    - Implement `create_dataset` method
    - Implement `to_json` and `from_json` methods
    - _Requirements: 1.5, 7.3, 7.4_
  - [x] 2.4 Write property test for dataset serialization round-trip
    - **Property 12: Dataset Serialization Round-Trip**
    - **Validates: Requirements 7.3, 7.4, 7.5**

- [x] 3. Implement SparseEncoder class (KC layer)
  - [x] 3.1 Create SparseEncoder with kWTA mechanism
    - Implement `__init__` with n_input, n_output, sparsity, connectivity
    - Implement `_initialize_weights` for sparse random connections
    - Implement `encode` method with k-Winner-Take-All
    - Implement `get_active_indices` method
    - _Requirements: 2.1, 2.2, 2.4_
  - [x] 3.2 Write property test for KC sparsity invariant
    - **Property 2: KC Sparsity Invariant**
    - **Validates: Requirements 2.2**
  - [x] 3.3 Write property test for pattern separation
    - **Property 3: Pattern Separation**
    - **Validates: Requirements 2.3**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement DrosophilaOlfactoryModel core
  - [x] 5.1 Create model initialization and prediction
    - Implement `__init__` with all parameters
    - Initialize W_pn_kc (fixed) and W_kc_mbon (plastic) matrices
    - Implement `predict` method returning (output, kc_activation)
    - _Requirements: 2.1, 3.1, 3.2_
  - [x] 5.2 Write property test for MBON output computation
    - **Property 4: MBON Output Computation**
    - **Validates: Requirements 3.1**
  - [x] 5.3 Implement modulation and learning methods
    - Implement `modulate` method with three-factor rule
    - Implement weight clipping to [0, 1]
    - Implement `train_aversive` and `train_appetitive` methods
    - Implement `reset_weights` method
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 3.5_
  - [x] 5.4 Write property test for weight bounds invariant
    - **Property 5: Weight Bounds Invariant**
    - **Validates: Requirements 3.5**
  - [x] 5.5 Write property test for three-factor learning rule
    - **Property 6: Three-Factor Learning Rule**
    - **Validates: Requirements 4.1, 4.4, 4.5**
  - [x] 5.6 Write property test for aversive learning (LTD)
    - **Property 7: Aversive Learning Causes LTD**
    - **Validates: Requirements 4.2**
  - [x] 5.7 Write property test for appetitive learning (LTP)
    - **Property 8: Appetitive Learning Causes LTP**
    - **Validates: Requirements 4.3**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement learning behavior verification
  - [x] 7.1 Implement complete training workflow
    - Wire together prediction and modulation
    - Verify training reduces response to trained odor
    - _Requirements: 5.1, 5.2, 5.4_
  - [x] 7.2 Write property test for learning effect
    - **Property 9: Learning Reduces Trained Odor Response**
    - **Validates: Requirements 5.2, 5.4**
  - [x] 7.3 Write property test for learning specificity
    - **Property 10: Learning Specificity**
    - **Validates: Requirements 5.3**

- [x] 8. Implement model serialization
  - [x] 8.1 Add JSON serialization to model
    - Implement `to_json` method for model state
    - Implement `from_json` class method for restoration
    - Handle numpy array serialization
    - _Requirements: 7.1, 7.2_
  - [x] 8.2 Write property test for model serialization round-trip
    - **Property 11: Model Serialization Round-Trip**
    - **Validates: Requirements 7.1, 7.2, 7.5**

- [x] 9. Implement ModelEvaluator class
  - [x] 9.1 Create evaluation methods
    - Implement `compute_discrimination_index`
    - Implement `compute_pattern_separation`
    - Implement `evaluate_generalization`
    - Implement `evaluate_specificity`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [x] 9.2 Write unit tests for evaluator methods
    - Test discrimination index calculation
    - Test pattern separation measurement
    - _Requirements: 6.1, 6.2_

- [x] 10. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
