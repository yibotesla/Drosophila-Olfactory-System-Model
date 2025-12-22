# Requirements Document

## Introduction

本项目旨在构建一个果蝇（*Drosophila*）嗅觉系统的计算模型，模拟从感知信息编码到行为决策输出的完整神经回路。该模型包含三层前馈网络架构（触角叶 AL → 蘑菇体 MB → 输出神经元 MBONs），并引入调制神经元（多巴胺神经元 DANs）实现联想学习。模型将展示稀疏编码、模式分离、三因素学习律等关键神经计算原理。

## Glossary

- **AL (Antennal Lobe)**: 触角叶，果蝇嗅觉系统的第一级处理中心，包含嗅小球
- **Glomeruli**: 嗅小球，AL 中接收嗅觉受体神经元信号的功能单元
- **PN (Projection Neuron)**: 投射神经元，将 AL 信号传递到蘑菇体
- **MB (Mushroom Body)**: 蘑菇体，果蝇大脑中负责学习和记忆的关键结构
- **KC (Kenyon Cell)**: Kenyon 细胞，蘑菇体中的主要神经元类型，约 2000 个
- **MBON (Mushroom Body Output Neuron)**: 蘑菇体输出神经元，整合 KC 信号并驱动行为
- **DAN (Dopaminergic Neuron)**: 多巴胺神经元，提供奖励/惩罚调制信号
- **kWTA (k-Winner-Take-All)**: 一种稀疏化机制，只保留前 k 个最强激活的神经元
- **LTD (Long-Term Depression)**: 长程抑制，突触权重的持久性降低
- **LTP (Long-Term Potentiation)**: 长程增强，突触权重的持久性增加
- **CS (Conditioned Stimulus)**: 条件刺激，如特定气味
- **US (Unconditioned Stimulus)**: 非条件刺激，如电击或食物奖励
- **Pattern Separation**: 模式分离，将相似输入映射到不同表征的过程
- **Three-Factor Learning Rule**: 三因素学习律，结合突触前活性、突触后活性和调制信号的学习规则

## Requirements

### Requirement 1: 气味输入表征

**User Story:** As a 研究者, I want to 将气味表示为高维特征向量, so that 模型能够处理和区分不同的嗅觉刺激。

#### Acceptance Criteria

1. WHEN 气味数据被输入到模型 THEN the System SHALL 将气味表示为 N 维向量，其中 N 等于嗅小球数量（默认 50）
2. WHEN 生成气味原型向量 THEN the System SHALL 确保向量元素取值范围在 [0, 1] 之间
3. WHEN 生成气味变体样本 THEN the System SHALL 支持添加高斯噪声以模拟生物噪声
4. WHEN 生成不同浓度的气味样本 THEN the System SHALL 支持通过浓度因子缩放原型向量
5. WHEN 生成训练数据集 THEN the System SHALL 能够创建包含多个气味原型及其变体的完整数据集

### Requirement 2: 稀疏编码层（蘑菇体 KC 层）

**User Story:** As a 研究者, I want to 实现 Kenyon Cell 层的稀疏编码机制, so that 模型能够进行模式分离并提高气味辨别能力。

#### Acceptance Criteria

1. WHEN 气味信号从 PN 层传递到 KC 层 THEN the System SHALL 使用随机稀疏连接矩阵 W_PN_KC 进行投影
2. WHEN KC 层接收到投影信号 THEN the System SHALL 应用 k-Winner-Take-All 机制，仅保留约 5% 的 KC 激活
3. WHEN 两个相似气味（重叠度 > 80%）经过 KC 层编码 THEN the System SHALL 产生显著不同的稀疏表征（模式分离）
4. WHEN 初始化 PN-KC 连接矩阵 THEN the System SHALL 生成稀疏随机连接，每个 KC 平均连接约 7 个 PN

### Requirement 3: 输出层与行为决策

**User Story:** As a 研究者, I want to 实现 MBON 输出层, so that 模型能够基于学习到的关联产生趋向或回避行为。

#### Acceptance Criteria

1. WHEN KC 层信号传递到 MBON THEN the System SHALL 计算加权和 y = Σ(w_i × KC_i) 作为输出
2. WHEN 初始化 KC-MBON 权重矩阵 THEN the System SHALL 将所有权重设置为 1.0
3. WHEN MBON 输出值较高 THEN the System SHALL 表示对该气味的趋向行为
4. WHEN MBON 输出值较低 THEN the System SHALL 表示对该气味的回避行为
5. WHEN 权重更新后 THEN the System SHALL 将权重限制在 [0, 1] 范围内

### Requirement 4: 调制神经元与三因素学习

**User Story:** As a 研究者, I want to 实现调制神经元（DANs）的学习机制, so that 模型能够通过奖励/惩罚信号进行联想学习。

#### Acceptance Criteria

1. WHEN 调制信号与气味同时出现 THEN the System SHALL 根据三因素学习律更新 KC-MBON 突触权重：Δw_i = η × KC_i × R
2. WHEN 厌恶刺激（正调制信号）与气味配对 THEN the System SHALL 降低对应 KC-MBON 连接的权重（LTD）
3. WHEN 奖励刺激（负调制信号）与气味配对 THEN the System SHALL 增加对应 KC-MBON 连接的权重（LTP）
4. WHEN 执行学习更新 THEN the System SHALL 仅修改当前活跃 KC 对应的突触权重
5. WHEN 学习率参数被设置 THEN the System SHALL 使用该参数控制权重更新的幅度

### Requirement 5: 联想学习训练流程

**User Story:** As a 研究者, I want to 执行完整的联想学习训练流程, so that 模型能够学会对特定气味产生回避或趋向行为。

#### Acceptance Criteria

1. WHEN 执行厌恶学习训练 THEN the System SHALL 在气味输入的同时提供正调制信号
2. WHEN 训练完成后再次呈现相同气味 THEN the System SHALL 产生降低的 MBON 输出（回避行为）
3. WHEN 训练完成后呈现未训练的气味 THEN the System SHALL 保持原有的 MBON 输出水平
4. WHEN 执行单次训练（One-shot Learning） THEN the System SHALL 能够产生可测量的行为改变

### Requirement 6: 模型评估与测试

**User Story:** As a 研究者, I want to 评估模型的学习性能和泛化能力, so that 我能够验证模型的生物学合理性。

#### Acceptance Criteria

1. WHEN 评估模式分离能力 THEN the System SHALL 计算相似气味在 KC 层表征的欧氏距离变化
2. WHEN 评估学习效果 THEN the System SHALL 计算训练前后 MBON 输出的差异（区分指数）
3. WHEN 评估泛化能力 THEN the System SHALL 测试模型对训练气味的噪声变体的响应
4. WHEN 评估特异性 THEN the System SHALL 验证未训练气味的响应不受显著影响

### Requirement 7: 数据序列化与模型持久化

**User Story:** As a 研究者, I want to 保存和加载模型状态及数据集, so that 我能够复现实验结果并进行长期研究。

#### Acceptance Criteria

1. WHEN 保存模型状态 THEN the System SHALL 将权重矩阵和参数序列化为 JSON 格式
2. WHEN 加载模型状态 THEN the System SHALL 从 JSON 文件恢复完整的模型配置
3. WHEN 保存数据集 THEN the System SHALL 将气味向量和标签序列化为可读格式
4. WHEN 加载数据集 THEN the System SHALL 正确解析并恢复原始数据结构
5. WHEN 序列化后再反序列化 THEN the System SHALL 产生与原始对象等价的结果
