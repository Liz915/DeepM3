import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Non-autonomous ODE: dz/dt = f(z, t, dt)
    
    Takes three signals:
    - z: hidden state
    - t: absolute time (cumulative log-hours) — "where in time"
    - dt: inter-event gap (log-hours) — "how long since last event"
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Time conditioning: absolute time + delta
        self.time_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [t_abs, dt] → D
            nn.SiLU(),
        )
        
        # Main dynamics: [z, time_cond] → dz/dt
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, t, z, dt=None):
        """
        t:  [B, 1] absolute time
        z:  [B, D] hidden state
        dt: [B, 1] inter-event delta (optional, defaults to 0)
        """
        # Normalize t to [B, 1]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.dim() > 2:
            t = t.view(z.size(0), -1)[:, :1]
        
        # Default dt to zeros if not provided
        if dt is None:
            dt = torch.zeros_like(t)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        
        # Combine absolute time + delta for richer temporal signal
        t_input = torch.cat([t, dt], dim=-1)  # [B, 2]
        t_emb = self.time_embed(t_input)       # [B, D]
        
        return self.net(torch.cat([z, t_emb], dim=-1))