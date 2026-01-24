import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    定义潜在状态的导数: dz/dt = f(z, t)
    模拟用户兴趣在连续时间上的漂移场。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 使用 MLP 拟合向量场
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh() # 限制导数范围，利于数值稳定
        )

    def forward(self, t, z):
        # 自治系统 (Autonomous System): 导数只与当前状态 z 有关，与绝对时间 t 无关
        # 但接口保留 t 以兼容非自治系统的扩展
        return self.net(z)