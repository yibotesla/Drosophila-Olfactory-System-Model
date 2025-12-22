# Requirements Document

## Introduction

本项目旨在对现有的果蝇嗅觉系统模型进行全面改进，解决已识别的关键缺陷、设计问题和功能缺失。改进涵盖：权重初始化和恢复的正确性、随机状态管理、权重更新规则的生物学合理性、多 MBON 支持、输入验证、学习历史记录、序列化完整性、配置管理等方面。

## Glossary

- **RNG (Random Number Generator)**: 随机数生成器，用于生成伪随机数序列
- **Generator**: NumPy 的新式随机数生成器接口，避免全局状态污染
- **LTD (Long-Term Depression)**: 长程抑制，突触权重的持久性降低
- **LTP (Long-Term Potentiation)**: 长程增强，突触权重的持久性增加
- **Multiplicative Learning Rule**: 乘法学习规则，权重更新与当前权重值成比例
- **Round-Trip**: 往返测试，序列化后反序列化应产生等价对象
- **NaN**: Not a Number，表示未定义或不可表示的数值
- **Inf**: Infinity，表示无穷大的数值

## Requirements

### Requirement 1: 修复权重初始化和恢复问题

**User Story:** As a 研究者, I want to 正确保存和恢复模型状态, so that 我能够精确复现实验结果。

#### Acceptance Criteria

1. WHEN 从 JSON 恢复模型状态 THEN the System SHALL 验证 PN-KC 权重矩阵的形状与模型参数匹配
2. WHEN PN-KC 权重矩阵形状不匹配 THEN the System SHALL 抛出 ValueError 并提供详细错误信息
3. WHEN 序列化模型状态 THEN the System SHALL 包含 seed 参数以支持完整恢复
4. WHEN 恢复模型状态 THEN the System SHALL 正确设置 W_kc_mbon_initial 为恢复的权重副本

### Requirement 2: 修复随机状态管理

**User Story:** As a 研究者, I want to 使用隔离的随机数生成器, so that 多个模型实例不会相互干扰随机状态。

#### Acceptance Criteria

1. WHEN 初始化 SparseEncoder THEN the System SHALL 使用 numpy.random.Generator 而非全局 np.random.seed
2. WHEN 提供 seed 参数 THEN the System SHALL 创建独立的 PCG64 随机数生成器
3. WHEN 未提供 seed 参数 THEN the System SHALL 使用 np.random.default_rng() 创建新生成器
4. WHEN 多个模型同时创建 THEN the System SHALL 确保各模型的随机状态相互独立

### Requirement 3: 改进权重更新规则

**User Story:** As a 研究者, I want to 使用更符合生物学的权重更新规则, so that 模型行为更接近真实神经系统。

#### Acceptance Criteria

1. WHEN 执行厌恶学习（LTD） THEN the System SHALL 使用乘法规则 w = w × (1 - α × R)
2. WHEN 执行奖励学习（LTP） THEN the System SHALL 使用乘法规则 w = w + α × |R| × (1 - w)
3. WHEN 权重更新后 THEN the System SHALL 将权重限制在 [0, 1] 范围内
4. WHEN 权重接近边界值 THEN the System SHALL 自然减缓更新速度（乘法规则的固有特性）

### Requirement 4: 增强输入验证

**User Story:** As a 研究者, I want to 获得清晰的输入验证错误信息, so that 我能够快速定位和修复数据问题。

#### Acceptance Criteria

1. WHEN 输入不是 numpy.ndarray 类型 THEN the System SHALL 抛出 TypeError 并说明期望类型
2. WHEN 输入维度不是一维 THEN the System SHALL 抛出 ValueError 并说明期望形状
3. WHEN 输入维度与 n_pn 不匹配 THEN the System SHALL 抛出 ValueError 并说明维度不匹配
4. WHEN 输入包含 NaN 或 Inf 值 THEN the System SHALL 抛出 ValueError 并说明无效值

### Requirement 5: 添加学习历史记录

**User Story:** As a 研究者, I want to 追踪模型的学习过程, so that 我能够分析学习曲线和权重变化。

#### Acceptance Criteria

1. WHEN 模型初始化 THEN the System SHALL 创建空的学习历史记录结构
2. WHEN 执行训练操作 THEN the System SHALL 记录训练类型、气味哈希、强度和权重变化
3. WHEN 请求学习历史 THEN the System SHALL 返回包含所有训练事件的完整记录
4. WHEN 重置模型权重 THEN the System SHALL 提供选项是否同时清除学习历史

### Requirement 6: 完善序列化功能

**User Story:** As a 研究者, I want to 完整保存和恢复模型的所有状态, so that 实验可以精确复现。

#### Acceptance Criteria

1. WHEN 序列化模型 THEN the System SHALL 包含 seed 参数
2. WHEN 序列化模型 THEN the System SHALL 使用 float64 精度保存权重矩阵
3. WHEN 反序列化模型 THEN the System SHALL 验证所有必需字段存在
4. WHEN 反序列化模型 THEN the System SHALL 验证权重矩阵维度与参数一致

### Requirement 7: 添加配置管理

**User Story:** As a 研究者, I want to 使用配置对象管理模型参数, so that 我能够方便地进行超参数搜索。

#### Acceptance Criteria

1. WHEN 创建 ModelConfig THEN the System SHALL 提供所有模型参数的默认值
2. WHEN 验证配置 THEN the System SHALL 检查所有参数在有效范围内
3. WHEN 配置参数无效 THEN the System SHALL 抛出 ValueError 并说明具体问题
4. WHEN 使用配置创建模型 THEN the System SHALL 正确应用所有配置参数

### Requirement 8: 支持多 MBON

**User Story:** As a 研究者, I want to 使用多个 MBON 输出神经元, so that 我能够模拟更复杂的行为决策。

#### Acceptance Criteria

1. WHEN 创建多 MBON 模型 THEN the System SHALL 正确初始化 (n_kc, n_mbon) 形状的权重矩阵
2. WHEN 计算区分指数 THEN the System SHALL 支持指定特定 MBON 索引
3. WHEN MBON 索引超出范围 THEN the System SHALL 抛出 ValueError
4. WHEN 评估泛化能力 THEN the System SHALL 返回所有 MBON 的响应

