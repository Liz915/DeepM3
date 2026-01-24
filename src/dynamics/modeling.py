import torch
import torch.nn as nn
# 确保引用路径正确
from src.dynamics.solver import ODESolver
from src.dynamics.func import ODEFunc

class DeepM3Model(nn.Module):
    # 这里的 __init__ 必须接收 solver 参数，否则 train.py 会报错
    def __init__(self, config, n_items=None, solver=None):
        super().__init__()
        
        # 1. 维度配置
        input_dim = config.get('input_dim', 384) 
        hidden_dim = config['model']['hidden_dim']
        
        # 优先使用传入的 solver (来自命令行)，否则读取配置文件
        # 这就是消融实验能跑通的关键！
        if solver is not None:
            self.solver_mode = solver
        else:
            self.solver_mode = config['model'].get('ode_solver', 'rk4')
            
        # 2. Embedding / Encoder 层
        if n_items is not None:
            self.use_embedding = True
            # 强制 Embedding 输出维度等于 hidden_dim
            self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)
            self.encoder = nn.Identity()
        else:
            self.use_embedding = False
            self.item_emb = None
            self.encoder = nn.Linear(input_dim, hidden_dim)

        # 3. 动力学核心
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 初始化 ODE 函数和求解器
        self.ode_func = ODEFunc(hidden_dim)
        self.solver = ODESolver(self.ode_func)
        
        # 4. 输出头
        output_dim = n_items if n_items is not None else 1
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t, **kwargs):
        # 1. Projection
        if self.use_embedding:
            x_emb = self.item_emb(x)
        else:
            x_emb = self.encoder(x)
            
        batch_size, seq_len, _ = x_emb.shape
        h = torch.zeros(batch_size, self.gru_cell.hidden_size).to(x.device)
        
        # 2. Loop (ODE-RNN 架构)
        for i in range(seq_len):
            # A. 离散更新 (观测点)
            current_input = x_emb[:, i, :]
            h = self.gru_cell(current_input, h)
            
            # B. 连续演化 (观测点之间)
            # 只有当不是最后一个点，且 solver_mode 不为 'none' 时才演化
            if i < seq_len - 1:
                if self.solver_mode != 'none':
                    # [Defense] 强制时间单调性，防止 dt <= 0
                    t_curr = t[:, i]
                    t_next = torch.maximum(t[:, i+1], t_curr + 1e-5) 
                    
                    dt = (t_next - t_curr).view(-1, 1)
                    
                    if self.solver_mode == 'rk4':
                        t_curr_expanded = t_curr.view(-1, 1)
                        # 调用 solver 的 rk4 方法
                        h = self.solver.rk4_step(h, t_curr_expanded, dt)
                    elif self.solver_mode == 'euler':
                        # 调用 solver 的 euler 方法 (或者手动写也行)
                        h = self.solver.euler_step(h, t_curr, dt)
        
        logits = self.head(h)
        return logits