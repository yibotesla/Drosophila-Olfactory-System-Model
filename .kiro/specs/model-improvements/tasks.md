# Implementation Plan

- [x] 1. 创建 ModelConfig 配置类
  - [x] 1.1 实现 ModelConfig dataclass
    - 在 `src/config.py` 创建 ModelConfig 数据类
    - 包含所有模型参数及默认值
    - 实现 `validate()` 方法检查参数有效性
    - _Requirements: 7.1, 7.2, 7.3_
  - [x] 1.2 编写 ModelConfig 验证属性测试
    - **Property 11: Config Validation**
    - **Validates: Requirements 7.2, 7.3**
  - [x] 1.3 编写 ModelConfig 应用属性测试
    - **Property 12: Config Application**
    - **Validates: Requirements 7.4**

- [x] 2. 修复 SparseEncoder 随机状态管理
  - [x] 2.1 重构 SparseEncoder 使用独立 RNG
    - 使用 `numpy.random.Generator` 和 `PCG64`
    - 移除全局 `np.random.seed()` 调用
    - 存储 RNG 实例为成员变量
    - _Requirements: 2.1, 2.2, 2.3_
  - [x] 2.2 编写随机状态隔离属性测试
    - **Property 4: Random State Isolation**
    - **Validates: Requirements 2.1, 2.4**
  - [x] 2.3 编写种子可重现性属性测试
    - **Property 5: Seed Reproducibility**
    - **Validates: Requirements 2.2**

- [x] 3. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 4. 增强输入验证
  - [x] 4.1 实现 _validate_input 方法
    - 在 DrosophilaOlfactoryModel 添加输入验证方法
    - 检查类型 (np.ndarray)、维度 (1D)、NaN/Inf
    - 在 predict() 方法开头调用验证
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 4.2 编写 NaN/Inf 拒绝属性测试
    - **Property 9: NaN/Inf Input Rejection**
    - **Validates: Requirements 4.4**

- [x] 5. 实现乘法学习规则
  - [x] 5.1 重构 modulate 方法
    - 实现 LTD 乘法规则: `w × (1 - η × R)`
    - 实现 LTP 乘法规则: `w + η × |R| × (1 - w)`
    - 保持权重裁剪到 [0, 1]
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [x] 5.2 编写乘法 LTD 规则属性测试
    - **Property 6: Multiplicative LTD Rule**
    - **Validates: Requirements 3.1**
  - [x] 5.3 编写乘法 LTP 规则属性测试
    - **Property 7: Multiplicative LTP Rule**
    - **Validates: Requirements 3.2**
  - [x] 5.4 编写边界更新减速属性测试
    - **Property 8: Boundary Update Deceleration**
    - **Validates: Requirements 3.4**

- [x] 6. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 7. 添加学习历史记录
  - [x] 7.1 实现学习历史功能
    - 在模型初始化时创建 `_learning_history` 列表
    - 在 train_aversive/train_appetitive 中记录事件
    - 实现 `get_learning_history()` 方法
    - 更新 `reset_weights()` 支持 `clear_history` 参数
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [x] 7.2 编写学习历史完整性属性测试
    - **Property 10: Learning History Completeness**
    - **Validates: Requirements 5.2, 5.3**

- [x] 8. 完善序列化功能
  - [x] 8.1 改进 to_json 方法
    - 添加 seed 字段到序列化输出
    - 添加 learning_history 字段
    - 确保使用 float64 精度
    - _Requirements: 6.1, 6.2_
  - [x] 8.2 改进 from_json 方法
    - 添加权重矩阵维度验证
    - 正确设置 W_kc_mbon_initial
    - 恢复 learning_history
    - _Requirements: 1.1, 1.2, 1.4, 6.3, 6.4_
  - [x] 8.3 编写权重维度验证属性测试
    - **Property 1: Weight Matrix Dimension Validation**
    - **Validates: Requirements 1.1, 1.2, 6.4**
  - [x] 8.4 编写种子序列化属性测试
    - **Property 2: Seed Serialization Completeness**
    - **Validates: Requirements 1.3, 6.1**
  - [x] 8.5 编写初始权重恢复属性测试
    - **Property 3: Initial Weights Restoration**
    - **Validates: Requirements 1.4**

- [x] 9. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 10. 完善多 MBON 支持
  - [x] 10.1 改进 ModelEvaluator
    - 更新 compute_discrimination_index 支持 mbon_idx 参数
    - 更新 evaluate_generalization 返回所有 MBON 响应
    - 添加 MBON 索引越界检查
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 10.2 编写多 MBON 权重形状属性测试
    - **Property 13: Multi-MBON Weight Shape**
    - **Validates: Requirements 8.1**
  - [x] 10.3 编写多 MBON 泛化响应属性测试
    - **Property 14: Multi-MBON Generalization Response**
    - **Validates: Requirements 8.4**

- [x] 11. 集成 ModelConfig 到模型
  - [x] 11.1 添加从 ModelConfig 创建模型的支持
    - 添加 `from_config` 类方法
    - 在 to_json 中包含完整配置
    - _Requirements: 7.4_

- [x] 12. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。
