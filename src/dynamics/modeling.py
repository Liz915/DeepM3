import torch
import torch.nn as nn
from src.dynamics.solver import ODESolver
from src.dynamics.func import ODEFunc

class DeepM3Model(nn.Module):
    def __init__(self, config, n_items=None):
        super().__init__()
        
        # 1. 维度配置 (Safety Fix)
        # 不管 input_dim 是多少，进入动力学层之前都映射到 hidden_dim
        input_dim = config.get('input_dim', 384) 
        hidden_dim = config['model']['hidden_dim']
        
        # 保存 solver_mode 用于 JIT 导出时的控制
        self.solver_mode = config['model'].get('ode_solver', 'rk4')
        
        # 2. Embedding / Encoder 层
        if n_items is not None:
            self.use_embedding = True
            # [Fix] 关键修改：Embedding 的输出维度强制设为 hidden_dim
            self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)
            self.encoder = nn.Identity()
        else:
            self.use_embedding = False
            self.item_emb = None
            # [Fix] 关键修改：Linear 将 input_dim 映射到 hidden_dim
            self.encoder = nn.Linear(input_dim, hidden_dim)

        # 3. 动力学核心 (现在输入输出都是 hidden_dim，绝对安全)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
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
        
        # 2. Loop
        for i in range(seq_len):
            # A. GRU Step
            current_input = x_emb[:, i, :]
            h = self.gru_cell(current_input, h)
            
            # B. ODE Step (JIT Friendly)
            if i < seq_len - 1:
                if self.solver_mode != 'none':
                    # [Fix] 强制时间单调性 (Model-level Defense)
                    # 防止这一步的时间比上一步小，导致 dt < 0
                    t_curr = t[:, i]
                    t_next = torch.maximum(t[:, i+1], t_curr + 1e-5) # 确保至少有微小增量
                    
                    dt = (t_next - t_curr).view(-1, 1)
                    
                    if self.solver_mode == 'rk4':
                        t_curr_expanded = t_curr.view(-1, 1)
                        h = self.solver.rk4_step(h, t_curr_expanded, dt)
                    elif self.solver_mode == 'euler':
                        dh = self.ode_func(t_curr, h)
                        h = h + dh * dt
        
        logits = self.head(h)
        return logits