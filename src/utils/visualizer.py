import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os

def plot_phase_portrait(hidden_states, times, save_path="assets/attractor.png"):
    """
    将高维的隐状态轨迹投影到 2D 相平面，展示动力学吸引子。
    
    hidden_states: [seq_len, hidden_dim] (取单个样本的轨迹)
    times: [seq_len] 时间戳
    """
    # 1. 使用 PCA 提取主成分 (Principal Component Analysis)
    # 我们想看从乱序中涌现出的"主结构" (Major Dynamics)
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(hidden_states.detach().cpu().numpy())
    
    # 2. 绘图
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 画轨迹线 (Trajectory)
    # 使用颜色渐变表示时间流逝 (Time Evolution)
    # 颜色越深，时间越晚 -> 可以看到兴趣是如何"流"向某个区域的
    scatter = ax.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                         c=times.detach().cpu().numpy(), 
                         cmap='viridis', s=100, alpha=0.8, edgecolors='w')
    
    ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 
            c='gray', alpha=0.3, linestyle='--', linewidth=1)
    
    # 3. 装饰图表 (Math Style)
    plt.title("Latent Dynamics Phase Portrait (Neural ODE)", fontsize=16)
    plt.xlabel("Principal Component 1 (Trend)", fontsize=12)
    plt.ylabel("Principal Component 2 (Cycle)", fontsize=12)
    plt.colorbar(scatter, label="Time Evolution")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 4. 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"🎨 Visualization saved to: {save_path}")
    plt.close()