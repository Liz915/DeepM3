import torch
import torch.nn as nn

class ODESolver(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def rk4_step(self, z0, t0, dt):
        """
        z0: [batch, dim]
        t0: [batch, 1]  <-- 必须保持二维
        dt: [batch, 1]
        """
        # k1
        k1 = self.func(t0, z0)
        
        # k2: t0 + dt/2
        k2 = self.func(t0 + dt * 0.5, z0 + dt * k1 * 0.5)
        
        # k3: t0 + dt/2
        k3 = self.func(t0 + dt * 0.5, z0 + dt * k2 * 0.5)
        
        # k4: t0 + dt
        k4 = self.func(t0 + dt, z0 + dt * k3)
        
        return z0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def forward(self, z0, t_span):
        """
        t_span: [batch, seq_len]
        """
        batch_size, seq_len = t_span.shape
        z_t = [z0]
        curr_z = z0
        
        for i in range(seq_len - 1):
            t_curr = t_span[:, i].unsqueeze(1)
            t_next = t_span[:, i+1].unsqueeze(1)
            dt = t_next - t_curr 
            
            # 这里的 dt 已经是 [B, 1] 了
            curr_z = self.rk4_step(curr_z, t_curr, dt)
            z_t.append(curr_z)
            
        return torch.stack(z_t, dim=1)