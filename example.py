#!/usr/bin/env python3
"""
示例脚本 - 演示果蝇嗅觉系统模型的基本使用
Example script demonstrating basic usage of the Drosophila Olfactory Model
"""

import numpy as np
from src.model import DrosophilaOlfactoryModel
from src.odor_dataset import OdorDataset
from src.evaluator import ModelEvaluator


def main():
    print("=" * 60)
    print("果蝇嗅觉系统计算模型 - 演示")
    print("Drosophila Olfactory System Model - Demo")
    print("=" * 60)
    
    # 1. 创建模型
    print("\n[1] 创建模型...")
    model = DrosophilaOlfactoryModel(
        n_pn=50,           # 50 个投射神经元
        n_kc=2000,         # 2000 个 Kenyon 细胞
        n_mbon=1,          # 1 个输出神经元
        sparsity=0.05,     # 5% 稀疏度
        learning_rate=0.05,
        seed=42
    )
    print(f"   - 投射神经元 (PN): {model.n_pn}")
    print(f"   - Kenyon 细胞 (KC): {model.n_kc}")
    print(f"   - 输出神经元 (MBON): {model.n_mbon}")
    print(f"   - KC 稀疏度: {model.sparsity * 100:.1f}%")
    
    # 2. 生成气味数据
    print("\n[2] 生成气味数据...")
    dataset = OdorDataset(n_features=50)
    
    # 生成两种气味原型
    odor_a = dataset.generate_prototype("Odor_A", seed=100)
    odor_b = dataset.generate_prototype("Odor_B", seed=200)
    
    print(f"   - 气味 A: 均值={odor_a.mean():.3f}, 标准差={odor_a.std():.3f}")
    print(f"   - 气味 B: 均值={odor_b.mean():.3f}, 标准差={odor_b.std():.3f}")
    
    # 3. 测试模式分离
    print("\n[3] 测试模式分离效果...")
    evaluator = ModelEvaluator(model)
    separation = evaluator.compute_pattern_separation(odor_a, odor_b)
    
    print(f"   - 输入空间距离: {separation['input_distance']:.4f}")
    print(f"   - KC 空间距离: {separation['kc_distance']:.4f}")
    print(f"   - 分离比: {separation['separation_ratio']:.4f}")
    print(f"   - 输入重叠度: {separation['input_overlap']:.4f}")
    print(f"   - KC 重叠度: {separation['kc_overlap']:.4f}")
    
    # 4. 厌恶学习演示
    print("\n[4] 厌恶学习演示...")
    
    # 训练前响应
    response_before_a, kc_a = model.predict(odor_a)
    response_before_b, _ = model.predict(odor_b)
    
    print(f"   训练前:")
    print(f"   - 气味 A 响应: {response_before_a[0]:.4f}")
    print(f"   - 气味 B 响应: {response_before_b[0]:.4f}")
    print(f"   - 气味 A 激活的 KC 数量: {int(kc_a.sum())}")
    
    # 对气味 A 进行厌恶训练
    print(f"\n   执行厌恶训练 (气味 A + 惩罚)...")
    weight_change = model.train_aversive(odor_a, strength=1.0)
    print(f"   - 权重变化量: {weight_change:.4f}")
    
    # 训练后响应
    response_after_a, _ = model.predict(odor_a)
    response_after_b, _ = model.predict(odor_b)
    
    print(f"\n   训练后:")
    print(f"   - 气味 A 响应: {response_after_a[0]:.4f} (变化: {response_after_a[0] - response_before_a[0]:.4f})")
    print(f"   - 气味 B 响应: {response_after_b[0]:.4f} (变化: {response_after_b[0] - response_before_b[0]:.4f})")
    
    # 计算区分指数
    di = evaluator.compute_discrimination_index(response_before_a[0], response_after_a[0])
    print(f"\n   区分指数 (气味 A): {di:.4f}")
    
    # 5. 学习特异性评估
    print("\n[5] 学习特异性评估...")
    specificity = evaluator.evaluate_specificity(odor_a, [odor_b])
    
    print(f"   - 训练气味响应: {specificity['trained_response']:.4f}")
    print(f"   - 未训练气味响应: {specificity['untrained_mean']:.4f}")
    print(f"   - 特异性指数: {specificity['specificity_index']:.4f}")
    
    # 6. 模型序列化演示
    print("\n[6] 模型序列化演示...")
    
    # 保存模型
    json_str = model.to_json()
    print(f"   - 序列化 JSON 长度: {len(json_str)} 字符")
    
    # 恢复模型
    restored_model = DrosophilaOlfactoryModel.from_json(json_str)
    
    # 验证恢复后的模型
    restored_response, _ = restored_model.predict(odor_a)
    print(f"   - 原模型响应: {response_after_a[0]:.4f}")
    print(f"   - 恢复模型响应: {restored_response[0]:.4f}")
    print(f"   - 响应一致: {np.allclose(response_after_a, restored_response)}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
