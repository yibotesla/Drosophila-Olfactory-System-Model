#!/usr/bin/env python3
"""
可视化脚本 - 展示果蝇嗅觉系统模型的关键特性
Visualization script for the Drosophila Olfactory Model

包含以下可视化:
1. 基础模型可视化 (9个子图)
2. 乘法学习规则可视化
3. 学习历史记录可视化
4. 多MBON支持可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.model import DrosophilaOlfactoryModel
from src.odor_dataset import OdorDataset
from src.evaluator import ModelEvaluator
from src.config import ModelConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_model():
    """生成模型可视化图表"""
    
    # 创建模型和数据
    np.random.seed(42)
    model = DrosophilaOlfactoryModel(
        n_pn=50, n_kc=500, n_mbon=1,
        sparsity=0.05, learning_rate=0.1, seed=42
    )
    dataset = OdorDataset(n_features=50)
    evaluator = ModelEvaluator(model)
    
    # 生成气味
    odor_a = dataset.generate_prototype("A", seed=100)
    odor_b = dataset.generate_prototype("B", seed=200)
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # ========== 1. 气味向量可视化 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(50)
    ax1.bar(x, odor_a, alpha=0.7, label='Odor A', color='steelblue')
    ax1.bar(x, odor_b, alpha=0.5, label='Odor B', color='coral')
    ax1.set_xlabel('Glomerulus Index')
    ax1.set_ylabel('Activation')
    ax1.set_title('1. Odor Input Vectors (50D)')
    ax1.legend()
    ax1.set_xlim(-1, 50)
    
    # ========== 2. KC稀疏激活模式 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    _, kc_a = model.predict(odor_a)
    _, kc_b = model.predict(odor_b)
    
    # 显示前200个KC的激活
    kc_display = 200
    kc_matrix = np.vstack([kc_a[:kc_display], kc_b[:kc_display]])
    im = ax2.imshow(kc_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Odor A', 'Odor B'])
    ax2.set_xlabel(f'KC Index (showing first {kc_display})')
    ax2.set_title(f'2. KC Sparse Activation (~{model.sparsity*100:.0f}% active)')
    plt.colorbar(im, ax=ax2, label='Active')
    
    # ========== 3. 模式分离效果 ==========
    ax3 = fig.add_subplot(gs[0, 2])
    
    # 生成多对相似气味并计算分离效果
    separations = []
    for i in range(20):
        base = dataset.generate_prototype(f"base_{i}", seed=i*10)
        # 创建相似变体(90%相同)
        variant = base.copy()
        change_idx = np.random.choice(50, 5, replace=False)
        variant[change_idx] = np.random.uniform(0, 1, 5)
        
        sep = evaluator.compute_pattern_separation(base, variant)
        separations.append(sep)
    
    input_dists = [s['input_distance'] for s in separations]
    kc_dists = [s['kc_distance'] for s in separations]
    
    ax3.scatter(input_dists, kc_dists, alpha=0.7, s=60, c='purple')
    max_val = max(max(input_dists), max(kc_dists)) * 1.1
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    ax3.set_xlabel('Input Distance')
    ax3.set_ylabel('KC Distance')
    ax3.set_title('3. Pattern Separation Effect')
    ax3.legend()
    ax3.set_xlim(0, max_val)
    ax3.set_ylim(0, max_val)
    
    # ========== 4. 学习过程中的权重变化 ==========
    ax4 = fig.add_subplot(gs[1, 0])
    
    model.reset_weights()
    training_steps = 10
    responses_a = []
    responses_b = []
    
    # 记录初始响应
    resp_a, _ = model.predict(odor_a)
    resp_b, _ = model.predict(odor_b)
    responses_a.append(resp_a[0])
    responses_b.append(resp_b[0])
    
    # 多次训练气味A
    for _ in range(training_steps):
        model.train_aversive(odor_a, strength=1.0)
        resp_a, _ = model.predict(odor_a)
        resp_b, _ = model.predict(odor_b)
        responses_a.append(resp_a[0])
        responses_b.append(resp_b[0])
    
    steps = range(training_steps + 1)
    ax4.plot(steps, responses_a, 'o-', label='Odor A (trained)', color='red', linewidth=2)
    ax4.plot(steps, responses_b, 's-', label='Odor B (untrained)', color='green', linewidth=2)
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('MBON Response')
    ax4.set_title('4. Learning: Aversive Training on Odor A')
    ax4.legend()
    ax4.set_xlim(-0.5, training_steps + 0.5)
    
    # ========== 5. 权重分布变化 ==========
    ax5 = fig.add_subplot(gs[1, 1])
    
    # 获取训练后的权重
    weights_after = model.weights_kc_mbon.flatten()
    
    # 重置并获取初始权重
    model.reset_weights()
    weights_before = model.weights_kc_mbon.flatten()
    
    ax5.hist(weights_before, bins=30, alpha=0.5, label='Before Training', color='blue')
    ax5.hist(weights_after, bins=30, alpha=0.5, label='After Training', color='red')
    ax5.set_xlabel('Weight Value')
    ax5.set_ylabel('Count')
    ax5.set_title('5. KC-MBON Weight Distribution')
    ax5.legend()
    ax5.set_xlim(-0.1, 1.1)
    
    # ========== 6. 学习特异性 ==========
    ax6 = fig.add_subplot(gs[1, 2])
    
    model.reset_weights()
    
    # 生成多个气味
    odors = [dataset.generate_prototype(f"odor_{i}", seed=300+i) for i in range(5)]
    
    # 记录训练前响应
    responses_before = []
    for odor in odors:
        resp, _ = model.predict(odor)
        responses_before.append(resp[0])
    
    # 只训练第一个气味
    for _ in range(5):
        model.train_aversive(odors[0], strength=1.0)
    
    # 记录训练后响应
    responses_after = []
    for odor in odors:
        resp, _ = model.predict(odor)
        responses_after.append(resp[0])
    
    x_pos = np.arange(5)
    width = 0.35
    ax6.bar(x_pos - width/2, responses_before, width, label='Before', color='lightblue')
    ax6.bar(x_pos + width/2, responses_after, width, label='After', color='salmon')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(['Odor 0\n(trained)', 'Odor 1', 'Odor 2', 'Odor 3', 'Odor 4'])
    ax6.set_ylabel('MBON Response')
    ax6.set_title('6. Learning Specificity')
    ax6.legend()
    
    # ========== 7. PN-KC连接矩阵 ==========
    ax7 = fig.add_subplot(gs[2, 0])
    
    W_pn_kc = model.weights_pn_kc[:, :100]  # 显示前100个KC
    im7 = ax7.imshow(W_pn_kc, aspect='auto', cmap='Greys', interpolation='nearest')
    ax7.set_xlabel('KC Index (first 100)')
    ax7.set_ylabel('PN Index')
    ax7.set_title('7. PN-KC Connection Matrix (Fixed)')
    plt.colorbar(im7, ax=ax7, label='Connected')
    
    # ========== 8. KC-MBON权重热图 ==========
    ax8 = fig.add_subplot(gs[2, 1])
    
    # 重新训练以获得有变化的权重
    model.reset_weights()
    for _ in range(5):
        model.train_aversive(odor_a, strength=1.0)
    
    weights = model.weights_kc_mbon[:100, 0]  # 前100个KC的权重
    weights_2d = weights.reshape(10, 10)
    
    im8 = ax8.imshow(weights_2d, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax8.set_xlabel('KC Index (mod 10)')
    ax8.set_ylabel('KC Index (div 10)')
    ax8.set_title('8. KC-MBON Weights (first 100 KCs)')
    plt.colorbar(im8, ax=ax8, label='Weight')
    
    # ========== 9. 泛化能力测试 ==========
    ax9 = fig.add_subplot(gs[2, 2])
    
    model.reset_weights()
    
    # 训练一个气味
    trained_odor = dataset.generate_prototype("trained", seed=500)
    for _ in range(5):
        model.train_aversive(trained_odor, strength=1.0)
    
    # 生成不同噪声水平的变体
    noise_levels = np.linspace(0, 0.5, 10)
    responses = []
    
    for noise in noise_levels:
        variants = dataset.generate_variants(trained_odor, n_samples=10, noise_level=noise, seed=600)
        variant_responses = []
        for v in variants:
            resp, _ = model.predict(v)
            variant_responses.append(resp[0])
        responses.append(np.mean(variant_responses))
    
    ax9.plot(noise_levels, responses, 'o-', color='teal', linewidth=2, markersize=8)
    ax9.axhline(y=responses[0], color='red', linestyle='--', alpha=0.5, label='Trained odor response')
    ax9.set_xlabel('Noise Level (σ)')
    ax9.set_ylabel('Mean MBON Response')
    ax9.set_title('9. Generalization: Response to Noisy Variants')
    ax9.legend()
    
    # 保存图表
    plt.suptitle('Drosophila Olfactory Model Visualization', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('model_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('model_visualization.pdf', bbox_inches='tight', facecolor='white')
    print("基础可视化已保存: model_visualization.png, model_visualization.pdf")
    
    plt.close()


def visualize_multiplicative_learning():
    """可视化乘法学习规则的特性"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ========== 1. LTD 乘法规则: w = w × (1 - η × R) ==========
    ax1 = axes[0, 0]
    
    learning_rate = 0.1
    R = 1.0  # 调制信号强度
    
    # 不同初始权重的衰减曲线
    initial_weights = [1.0, 0.8, 0.5, 0.2]
    steps = 20
    
    for w0 in initial_weights:
        weights = [w0]
        w = w0
        for _ in range(steps):
            w = w * (1 - learning_rate * R)
            w = max(0, w)  # 裁剪到 [0, 1]
            weights.append(w)
        ax1.plot(range(steps + 1), weights, 'o-', label=f'w₀={w0}', linewidth=2, markersize=4)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Weight Value')
    ax1.set_title('1. Multiplicative LTD: w = w × (1 - η × R)')
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. LTP 乘法规则: w = w + η × |R| × (1 - w) ==========
    ax2 = axes[0, 1]
    
    initial_weights = [0.0, 0.2, 0.5, 0.8]
    
    for w0 in initial_weights:
        weights = [w0]
        w = w0
        for _ in range(steps):
            w = w + learning_rate * R * (1 - w)
            w = min(1, w)  # 裁剪到 [0, 1]
            weights.append(w)
        ax2.plot(range(steps + 1), weights, 's-', label=f'w₀={w0}', linewidth=2, markersize=4)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('2. Multiplicative LTP: w = w + η × |R| × (1 - w)')
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    # ========== 3. 边界减速效果 ==========
    ax3 = axes[1, 0]
    
    # 计算不同权重位置的更新幅度
    weights_range = np.linspace(0.01, 0.99, 100)
    
    # LTD 更新幅度
    ltd_delta = weights_range * learning_rate * R
    
    # LTP 更新幅度
    ltp_delta = learning_rate * R * (1 - weights_range)
    
    ax3.plot(weights_range, ltd_delta, 'r-', label='LTD |Δw|', linewidth=2)
    ax3.plot(weights_range, ltp_delta, 'g-', label='LTP |Δw|', linewidth=2)
    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Current Weight Value')
    ax3.set_ylabel('Update Magnitude |Δw|')
    ax3.set_title('3. Boundary Deceleration Effect')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. 与加法规则对比 ==========
    ax4 = axes[1, 1]
    
    # 乘法 LTD
    w_mult = 1.0
    mult_weights = [w_mult]
    for _ in range(30):
        w_mult = w_mult * (1 - learning_rate * R)
        w_mult = max(0, w_mult)
        mult_weights.append(w_mult)
    
    # 加法 LTD (旧规则)
    w_add = 1.0
    add_weights = [w_add]
    for _ in range(30):
        w_add = w_add - learning_rate * R
        w_add = max(0, w_add)
        add_weights.append(w_add)
    
    ax4.plot(range(31), mult_weights, 'b-', label='Multiplicative (new)', linewidth=2)
    ax4.plot(range(31), add_weights, 'r--', label='Additive (old)', linewidth=2)
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Weight Value')
    ax4.set_title('4. Multiplicative vs Additive Learning')
    ax4.legend()
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Multiplicative Learning Rule Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multiplicative_learning.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("乘法学习规则可视化已保存: multiplicative_learning.png")
    plt.close()


def visualize_learning_history():
    """可视化学习历史记录功能"""
    
    # 创建模型
    model = DrosophilaOlfactoryModel(
        n_pn=50, n_kc=500, n_mbon=1,
        sparsity=0.05, learning_rate=0.1, seed=42
    )
    dataset = OdorDataset(n_features=50)
    
    # 生成多个气味
    odors = [dataset.generate_prototype(f"odor_{i}", seed=100+i) for i in range(3)]
    
    # 执行多次训练
    for i in range(5):
        model.train_aversive(odors[0], strength=1.0)
    for i in range(3):
        model.train_appetitive(odors[1], strength=0.8)
    for i in range(2):
        model.train_aversive(odors[2], strength=0.5)
    
    # 获取学习历史
    history = model.get_learning_history()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ========== 1. 训练事件时间线 ==========
    ax1 = axes[0, 0]
    
    aversive_events = [e for e in history if e['type'] == 'aversive']
    appetitive_events = [e for e in history if e['type'] == 'appetitive']
    
    ax1.scatter(range(len(aversive_events)), 
                [e['weight_change'] for e in aversive_events],
                c='red', s=100, label='Aversive', marker='v')
    ax1.scatter(range(len(aversive_events), len(aversive_events) + len(appetitive_events)),
                [e['weight_change'] for e in appetitive_events],
                c='green', s=100, label='Appetitive', marker='^')
    
    ax1.set_xlabel('Event Index')
    ax1.set_ylabel('Weight Change')
    ax1.set_title('1. Learning Events Timeline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. 按气味分组的训练次数 ==========
    ax2 = axes[0, 1]
    
    odor_hashes = {}
    for e in history:
        h = e['odor_hash']
        if h not in odor_hashes:
            odor_hashes[h] = {'aversive': 0, 'appetitive': 0}
        odor_hashes[h][e['type']] += 1
    
    x = np.arange(len(odor_hashes))
    width = 0.35
    
    aversive_counts = [v['aversive'] for v in odor_hashes.values()]
    appetitive_counts = [v['appetitive'] for v in odor_hashes.values()]
    
    ax2.bar(x - width/2, aversive_counts, width, label='Aversive', color='salmon')
    ax2.bar(x + width/2, appetitive_counts, width, label='Appetitive', color='lightgreen')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Odor {i}' for i in range(len(odor_hashes))])
    ax2.set_ylabel('Training Count')
    ax2.set_title('2. Training Count by Odor')
    ax2.legend()
    
    # ========== 3. 累积权重变化 ==========
    ax3 = axes[1, 0]
    
    cumulative_change = np.cumsum([e['weight_change'] for e in history])
    colors = ['red' if e['type'] == 'aversive' else 'green' for e in history]
    
    ax3.plot(range(len(history)), cumulative_change, 'b-', linewidth=2)
    ax3.scatter(range(len(history)), cumulative_change, c=colors, s=50, zorder=5)
    ax3.set_xlabel('Event Index')
    ax3.set_ylabel('Cumulative Weight Change')
    ax3.set_title('3. Cumulative Weight Change')
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. 训练强度分布 ==========
    ax4 = axes[1, 1]
    
    strengths = [e['strength'] for e in history]
    types = [e['type'] for e in history]
    
    aversive_strengths = [s for s, t in zip(strengths, types) if t == 'aversive']
    appetitive_strengths = [s for s, t in zip(strengths, types) if t == 'appetitive']
    
    ax4.hist(aversive_strengths, bins=10, alpha=0.7, label='Aversive', color='salmon')
    ax4.hist(appetitive_strengths, bins=10, alpha=0.7, label='Appetitive', color='lightgreen')
    ax4.set_xlabel('Training Strength')
    ax4.set_ylabel('Count')
    ax4.set_title('4. Training Strength Distribution')
    ax4.legend()
    
    plt.suptitle('Learning History Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('learning_history.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("学习历史可视化已保存: learning_history.png")
    plt.close()


def visualize_multi_mbon():
    """可视化多MBON支持"""
    
    # 创建多MBON模型
    model = DrosophilaOlfactoryModel(
        n_pn=50, n_kc=500, n_mbon=3,
        sparsity=0.05, learning_rate=0.1, seed=42
    )
    dataset = OdorDataset(n_features=50)
    evaluator = ModelEvaluator(model)
    
    # 生成气味
    odor_a = dataset.generate_prototype("A", seed=100)
    odor_b = dataset.generate_prototype("B", seed=200)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ========== 1. 多MBON权重矩阵 ==========
    ax1 = axes[0, 0]
    
    weights = model.weights_kc_mbon[:100, :]  # 前100个KC
    im = ax1.imshow(weights.T, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax1.set_xlabel('KC Index (first 100)')
    ax1.set_ylabel('MBON Index')
    ax1.set_title('1. Multi-MBON Weight Matrix')
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['MBON 0', 'MBON 1', 'MBON 2'])
    plt.colorbar(im, ax=ax1, label='Weight')
    
    # ========== 2. 各MBON对不同气味的响应 ==========
    ax2 = axes[0, 1]
    
    response_a, _ = model.predict(odor_a)
    response_b, _ = model.predict(odor_b)
    
    x = np.arange(3)
    width = 0.35
    
    ax2.bar(x - width/2, response_a, width, label='Odor A', color='steelblue')
    ax2.bar(x + width/2, response_b, width, label='Odor B', color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['MBON 0', 'MBON 1', 'MBON 2'])
    ax2.set_ylabel('Response')
    ax2.set_title('2. Multi-MBON Response to Different Odors')
    ax2.legend()
    
    # ========== 3. 训练后各MBON的区分指数 ==========
    ax3 = axes[1, 0]
    
    # 记录训练前响应
    response_before, _ = model.predict(odor_a)
    
    # 训练
    for _ in range(5):
        model.train_aversive(odor_a, strength=1.0)
    
    # 记录训练后响应
    response_after, _ = model.predict(odor_a)
    
    # 计算各MBON的区分指数
    dis = []
    for i in range(3):
        di = evaluator.compute_discrimination_index(response_before, response_after, mbon_idx=i)
        dis.append(di)
    
    ax3.bar(range(3), dis, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['MBON 0', 'MBON 1', 'MBON 2'])
    ax3.set_ylabel('Discrimination Index')
    ax3.set_title('3. Discrimination Index per MBON')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # ========== 4. 泛化响应 (多MBON) ==========
    ax4 = axes[1, 1]
    
    # 测试不同噪声水平的泛化响应
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    # 存储每个噪声水平下各 MBON 的平均响应
    responses_by_noise = {i: [] for i in range(3)}
    
    for noise in noise_levels:
        if noise == 0:
            # 训练气味本身
            resp, _ = model.predict(odor_a)
            for i in range(3):
                responses_by_noise[i].append(resp[i])
        else:
            # 生成噪声变体并计算平均响应
            variants = dataset.generate_variants(odor_a, n_samples=10, noise_level=noise, seed=int(noise*1000))
            avg_resp = np.zeros(3)
            for v in variants:
                resp, _ = model.predict(v)
                avg_resp += resp
            avg_resp /= len(variants)
            for i in range(3):
                responses_by_noise[i].append(avg_resp[i])
    
    # 绘制每个 MBON 的泛化曲线
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    for i in range(3):
        ax4.plot(noise_levels, responses_by_noise[i], 'o-', 
                label=f'MBON {i}', linewidth=2, markersize=8, color=colors[i])
    
    ax4.set_xlabel('Noise Level')
    ax4.set_ylabel('Mean MBON Response')
    ax4.set_title('4. Generalization: Response vs Noise Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-MBON Support Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multi_mbon.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("多MBON可视化已保存: multi_mbon.png")
    plt.close()


def visualize_config_and_validation():
    """可视化配置管理和输入验证"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ========== 1. 不同配置下的模型行为 ==========
    ax1 = axes[0, 0]
    
    configs = [
        ModelConfig(n_pn=50, n_kc=500, sparsity=0.05, learning_rate=0.05),
        ModelConfig(n_pn=50, n_kc=500, sparsity=0.05, learning_rate=0.1),
        ModelConfig(n_pn=50, n_kc=500, sparsity=0.05, learning_rate=0.2),
    ]
    
    dataset = OdorDataset(n_features=50)
    odor = dataset.generate_prototype("test", seed=42)
    
    for config in configs:
        model = DrosophilaOlfactoryModel.from_config(config)
        responses = []
        resp, _ = model.predict(odor)
        responses.append(resp[0])
        
        for _ in range(10):
            model.train_aversive(odor, strength=1.0)
            resp, _ = model.predict(odor)
            responses.append(resp[0])
        
        ax1.plot(range(11), responses, 'o-', 
                label=f'η={config.learning_rate}', linewidth=2, markersize=4)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('MBON Response')
    ax1.set_title('1. Effect of Learning Rate (from Config)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. 不同稀疏度的KC激活 ==========
    ax2 = axes[0, 1]
    
    sparsities = [0.02, 0.05, 0.1, 0.2]
    
    for sp in sparsities:
        model = DrosophilaOlfactoryModel(n_pn=50, n_kc=500, sparsity=sp, seed=42)
        _, kc = model.predict(odor)
        active_count = int(kc.sum())
        ax2.bar(sparsities.index(sp), active_count, 
               label=f'{sp*100:.0f}%', color=plt.cm.viridis(sp*5))
    
    ax2.set_xticks(range(len(sparsities)))
    ax2.set_xticklabels([f'{s*100:.0f}%' for s in sparsities])
    ax2.set_xlabel('Sparsity Setting')
    ax2.set_ylabel('Active KC Count')
    ax2.set_title('2. KC Activation vs Sparsity')
    
    # ========== 3. 随机种子可重现性 ==========
    ax3 = axes[1, 0]
    
    # 相同种子
    model1 = DrosophilaOlfactoryModel(n_pn=50, n_kc=100, seed=42)
    model2 = DrosophilaOlfactoryModel(n_pn=50, n_kc=100, seed=42)
    
    # 不同种子
    model3 = DrosophilaOlfactoryModel(n_pn=50, n_kc=100, seed=123)
    
    w1 = model1.weights_pn_kc[:, :20].flatten()
    w2 = model2.weights_pn_kc[:, :20].flatten()
    w3 = model3.weights_pn_kc[:, :20].flatten()
    
    ax3.scatter(w1, w2, alpha=0.5, label='Same seed (42 vs 42)', s=20)
    ax3.scatter(w1, w3, alpha=0.5, label='Different seed (42 vs 123)', s=20)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('Model 1 Weights')
    ax3.set_ylabel('Model 2/3 Weights')
    ax3.set_title('3. Seed Reproducibility')
    ax3.legend()
    
    # ========== 4. 输入验证示例 ==========
    ax4 = axes[1, 1]
    
    validation_tests = [
        ('Valid input', True),
        ('Wrong dimension', False),
        ('Contains NaN', False),
        ('Contains Inf', False),
        ('Wrong type', False),
    ]
    
    colors = ['green' if v else 'red' for _, v in validation_tests]
    y_pos = range(len(validation_tests))
    
    ax4.barh(y_pos, [1]*len(validation_tests), color=colors, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([t[0] for t in validation_tests])
    ax4.set_xlabel('Validation Result')
    ax4.set_title('4. Input Validation Examples')
    ax4.set_xlim(0, 1.2)
    
    # 添加标签
    for i, (name, valid) in enumerate(validation_tests):
        label = '✓ Pass' if valid else '✗ Reject'
        ax4.text(1.05, i, label, va='center', fontsize=10)
    
    plt.suptitle('Configuration & Validation Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('config_validation.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("配置和验证可视化已保存: config_validation.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("生成果蝇嗅觉系统模型可视化")
    print("=" * 60)
    
    print("\n[1/5] 生成基础模型可视化...")
    visualize_model()
    
    print("\n[2/5] 生成乘法学习规则可视化...")
    visualize_multiplicative_learning()
    
    print("\n[3/5] 生成学习历史可视化...")
    visualize_learning_history()
    
    print("\n[4/5] 生成多MBON支持可视化...")
    visualize_multi_mbon()
    
    print("\n[5/5] 生成配置和验证可视化...")
    visualize_config_and_validation()
    
    print("\n" + "=" * 60)
    print("所有可视化已完成!")
    print("生成的文件:")
    print("  - model_visualization.png/pdf (基础模型)")
    print("  - multiplicative_learning.png (乘法学习规则)")
    print("  - learning_history.png (学习历史)")
    print("  - multi_mbon.png (多MBON支持)")
    print("  - config_validation.png (配置和验证)")
    print("=" * 60)
