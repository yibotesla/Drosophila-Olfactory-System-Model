#!/usr/bin/env python3
"""
模型设计目的验证分析
Analyze whether the model achieves its design goals
"""

import numpy as np
from src.model import DrosophilaOlfactoryModel
from src.odor_dataset import OdorDataset
from src.evaluator import ModelEvaluator

def analyze_model():
    print("=" * 70)
    print("果蝇嗅觉系统模型 - 设计目的验证分析")
    print("Drosophila Olfactory Model - Design Goal Verification")
    print("=" * 70)
    
    # 创建模型
    model = DrosophilaOlfactoryModel(
        n_pn=50, n_kc=2000, n_mbon=1,
        sparsity=0.05, learning_rate=0.1, seed=42
    )
    dataset = OdorDataset(n_features=50)
    evaluator = ModelEvaluator(model)
    
    print("\n" + "=" * 70)
    print("1. 稀疏编码验证 (Sparse Coding)")
    print("=" * 70)
    
    odor = dataset.generate_prototype("test", seed=100)
    _, kc_activation = model.predict(odor)
    
    active_count = int(kc_activation.sum())
    expected_active = int(model.n_kc * model.sparsity)
    sparsity_actual = active_count / model.n_kc * 100
    
    print(f"   KC 总数: {model.n_kc}")
    print(f"   设定稀疏度: {model.sparsity * 100:.1f}%")
    print(f"   实际激活 KC 数: {active_count}")
    print(f"   期望激活 KC 数: {expected_active}")
    print(f"   实际稀疏度: {sparsity_actual:.2f}%")
    print(f"   ✅ 稀疏编码正常工作" if abs(active_count - expected_active) <= 1 else "   ❌ 稀疏编码异常")
    
    print("\n" + "=" * 70)
    print("2. 模式分离验证 (Pattern Separation)")
    print("=" * 70)
    
    # 生成多对相似气味
    separations = []
    for i in range(20):
        base = dataset.generate_prototype(f"base_{i}", seed=i*10)
        # 创建90%相似的变体
        variant = base.copy()
        change_idx = np.random.choice(50, 5, replace=False)
        variant[change_idx] = np.random.uniform(0, 1, 5)
        
        sep = evaluator.compute_pattern_separation(base, variant)
        separations.append(sep)
    
    avg_input_overlap = np.mean([s['input_overlap'] for s in separations])
    avg_kc_overlap = np.mean([s['kc_overlap'] for s in separations])
    avg_separation_ratio = np.mean([s['separation_ratio'] for s in separations])
    
    print(f"   测试样本数: 20 对相似气味")
    print(f"   平均输入重叠度: {avg_input_overlap:.4f}")
    print(f"   平均 KC 重叠度: {avg_kc_overlap:.4f}")
    print(f"   重叠度降低: {(avg_input_overlap - avg_kc_overlap) / avg_input_overlap * 100:.1f}%")
    print(f"   平均分离比: {avg_separation_ratio:.4f}")
    
    if avg_kc_overlap < avg_input_overlap:
        print(f"   ✅ 模式分离有效 (KC 重叠度 < 输入重叠度)")
    else:
        print(f"   ❌ 模式分离无效")
    
    print("\n" + "=" * 70)
    print("3. 联想学习验证 (Associative Learning)")
    print("=" * 70)
    
    model.reset_weights(clear_history=True)
    
    odor_a = dataset.generate_prototype("A", seed=100)
    odor_b = dataset.generate_prototype("B", seed=200)
    
    # 训练前响应
    response_before_a, _ = model.predict(odor_a)
    response_before_b, _ = model.predict(odor_b)
    
    print(f"\n   训练前响应:")
    print(f"   - 气味 A: {response_before_a[0]:.4f}")
    print(f"   - 气味 B: {response_before_b[0]:.4f}")
    
    # 对气味 A 进行厌恶训练
    print(f"\n   执行厌恶训练 (气味 A, 5次)...")
    for _ in range(5):
        model.train_aversive(odor_a, strength=1.0)
    
    # 训练后响应
    response_after_a, _ = model.predict(odor_a)
    response_after_b, _ = model.predict(odor_b)
    
    print(f"\n   训练后响应:")
    print(f"   - 气味 A: {response_after_a[0]:.4f} (变化: {response_after_a[0] - response_before_a[0]:.4f})")
    print(f"   - 气味 B: {response_after_b[0]:.4f} (变化: {response_after_b[0] - response_before_b[0]:.4f})")
    
    # 计算区分指数
    di_a = evaluator.compute_discrimination_index(response_before_a, response_after_a)
    di_b = evaluator.compute_discrimination_index(response_before_b, response_after_b)
    
    print(f"\n   区分指数:")
    print(f"   - 气味 A (训练): {di_a:.4f}")
    print(f"   - 气味 B (未训练): {di_b:.4f}")
    
    if response_after_a[0] < response_before_a[0]:
        print(f"   ✅ 厌恶学习有效 (训练气味响应降低)")
    else:
        print(f"   ❌ 厌恶学习无效")
    
    print("\n" + "=" * 70)
    print("4. 学习特异性验证 (Learning Specificity)")
    print("=" * 70)
    
    relative_change_b = abs(response_after_b[0] - response_before_b[0]) / response_before_b[0] * 100
    
    print(f"   未训练气味响应变化: {relative_change_b:.2f}%")
    
    if relative_change_b < 10:
        print(f"   ✅ 学习特异性良好 (未训练气味变化 < 10%)")
    else:
        print(f"   ⚠️ 学习特异性一般 (存在一定泛化)")
    
    print("\n" + "=" * 70)
    print("5. 乘法学习规则验证 (Multiplicative Learning)")
    print("=" * 70)
    
    model.reset_weights(clear_history=True)
    
    # 测试边界减速效果
    weights_history = []
    weights_history.append(model.weights_kc_mbon.mean())
    
    for i in range(20):
        model.train_aversive(odor_a, strength=1.0)
        weights_history.append(model.weights_kc_mbon.mean())
    
    # 计算前5次和后5次的平均变化
    early_changes = [weights_history[i] - weights_history[i+1] for i in range(5)]
    late_changes = [weights_history[i] - weights_history[i+1] for i in range(15, 20)]
    
    avg_early = np.mean(early_changes)
    avg_late = np.mean(late_changes)
    
    print(f"   前5次训练平均权重变化: {avg_early:.6f}")
    print(f"   后5次训练平均权重变化: {avg_late:.6f}")
    print(f"   变化减速比: {avg_early / avg_late:.2f}x")
    
    if avg_late < avg_early:
        print(f"   ✅ 乘法规则有效 (权重接近边界时更新减速)")
    else:
        print(f"   ❌ 乘法规则无效")
    
    print("\n" + "=" * 70)
    print("6. 学习历史记录验证 (Learning History)")
    print("=" * 70)
    
    history = model.get_learning_history()
    print(f"   记录的训练事件数: {len(history)}")
    print(f"   期望的训练事件数: 20")
    
    if len(history) == 20:
        print(f"   ✅ 学习历史记录完整")
        print(f"   最后一次训练:")
        print(f"   - 类型: {history[-1]['type']}")
        print(f"   - 强度: {history[-1]['strength']}")
        print(f"   - 权重变化: {history[-1]['weight_change']:.6f}")
    else:
        print(f"   ❌ 学习历史记录不完整")
    
    print("\n" + "=" * 70)
    print("7. 泛化能力验证 (Generalization)")
    print("=" * 70)
    
    model.reset_weights(clear_history=True)
    
    # 训练一个气味
    trained_odor = dataset.generate_prototype("trained", seed=500)
    for _ in range(5):
        model.train_aversive(trained_odor, strength=1.0)
    
    # 测试不同噪声水平的变体
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    responses = []
    
    for noise in noise_levels:
        if noise == 0:
            resp, _ = model.predict(trained_odor)
            responses.append(resp[0])
        else:
            variants = dataset.generate_variants(trained_odor, n_samples=10, noise_level=noise, seed=600)
            variant_responses = [model.predict(v)[0][0] for v in variants]
            responses.append(np.mean(variant_responses))
    
    print(f"   噪声水平 -> 平均响应:")
    for noise, resp in zip(noise_levels, responses):
        print(f"   - {noise:.1f}: {resp:.4f}")
    
    # 检查泛化梯度
    if responses[0] < responses[-1]:
        print(f"   ✅ 泛化梯度正常 (噪声增加 -> 响应恢复)")
    else:
        print(f"   ⚠️ 泛化梯度异常")
    
    print("\n" + "=" * 70)
    print("8. 序列化往返验证 (Serialization Round-Trip)")
    print("=" * 70)
    
    # 保存当前模型
    json_str = model.to_json()
    
    # 恢复模型
    restored_model = DrosophilaOlfactoryModel.from_json(json_str)
    
    # 验证
    original_response, _ = model.predict(trained_odor)
    restored_response, _ = restored_model.predict(trained_odor)
    
    weights_match = np.allclose(model.weights_kc_mbon, restored_model.weights_kc_mbon)
    response_match = np.allclose(original_response, restored_response)
    history_match = len(model.get_learning_history()) == len(restored_model.get_learning_history())
    
    print(f"   权重矩阵一致: {'✅' if weights_match else '❌'}")
    print(f"   响应输出一致: {'✅' if response_match else '❌'}")
    print(f"   学习历史一致: {'✅' if history_match else '❌'}")
    
    if weights_match and response_match and history_match:
        print(f"   ✅ 序列化往返验证通过")
    else:
        print(f"   ❌ 序列化往返验证失败")
    
    print("\n" + "=" * 70)
    print("总结 (Summary)")
    print("=" * 70)
    
    results = {
        "稀疏编码": abs(active_count - expected_active) <= 1,
        "模式分离": avg_kc_overlap < avg_input_overlap,
        "联想学习": response_after_a[0] < response_before_a[0],
        "学习特异性": relative_change_b < 15,
        "乘法学习规则": avg_late < avg_early,
        "学习历史记录": len(history) == 20,
        "泛化能力": responses[0] < responses[-1],
        "序列化往返": weights_match and response_match and history_match,
    }
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n   验证项目: {total}")
    print(f"   通过: {passed}")
    print(f"   失败: {total - passed}")
    print(f"\n   详细结果:")
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   - {name}: {status}")
    
    print(f"\n   总体评估: ", end="")
    if passed == total:
        print("🎉 模型完全达到设计目的!")
    elif passed >= total * 0.8:
        print("✅ 模型基本达到设计目的")
    elif passed >= total * 0.6:
        print("⚠️ 模型部分达到设计目的")
    else:
        print("❌ 模型未达到设计目的")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_model()
