import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    定义潜在状态的导数: dz/dt = f(z, t)
    模拟用户兴趣在连续时间上的漂移场 (Drift Field)。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 使用一个简单的 MLP 来拟合向量场
        # Tanh 激活函数通常在动力学建模中比 ReLU 更稳定 (有界性)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh() # 限制导数范围，防止梯度爆炸 (Lipschitz continuity constraint idea)
        )

    def forward(self, t, z):
        # 这里的 t 可以作为位置编码注入，但为了简化暂只用 autonomous system (自洽系统)
        return self.net(z)