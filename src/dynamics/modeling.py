import torch
import torch.nn as nn

from src.dynamics.solver import ODESolver
from src.dynamics.func import ODEFunc


class DeepM3Model(nn.Module):
    """
    Event-wise ODE-RNN (Rubanova et al., 2019).
    
    For each event in the sequence:
        1. ODE evolve: h = ODE(h, t_{i-1} → t_i)  -- continuous dynamics between events
        2. GRU update: h = GRUCell(h, x_emb_i)     -- discrete update on observation
    
    This makes EVERY timestamp matter, not just the last one.
    """
    def __init__(self, config, n_items=None, solver=None):
        super().__init__()

        hidden_dim = config['model']['hidden_dim']
        self.hidden_dim = hidden_dim

        # Solver mode
        if solver is not None:
            self.solver_mode = solver
        else:
            self.solver_mode = config['model'].get('ode_solver', 'rk4')

        # Item embedding
        if n_items is not None:
            self.use_embedding = True
            self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)
            nn.init.xavier_normal_(self.item_emb.weight)
        else:
            self.use_embedding = False
            self.item_emb = None
            input_dim = config.get('input_dim', 64)
            self.encoder = nn.Linear(input_dim, hidden_dim)

        # GRUCell for per-event discrete updates
        self.gru_cell = nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
        )
        
        # Also keep fused GRU for 'none' mode (baseline-equivalent, ignores time)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # ODE components
        self.ode_func = ODEFunc(hidden_dim)
        self.solver = ODESolver(self.ode_func)

    def forward(self, x, t, dt=None, **kwargs):
        """
        x:  [B, L] item IDs
        t:  [B, L] cumulative time (log-hours)
        dt: [B, L] inter-event deltas (log-hours), optional
        Returns: [B, D] user embedding
        """
        # 1. Item embeddings
        if self.use_embedding:
            x_emb = self.item_emb(x)  # [B, L, D]
        else:
            x_emb = self.encoder(x)

        B, L, D = x_emb.shape

        # 'none' mode: use fused GRU (no ODE, same as baseline)
        if self.solver_mode in ('none', 'baseline'):
            _, h_n = self.gru(x_emb)
            return h_n.squeeze(0)

        # Event-wise ODE-RNN loop
        if dt is None:
            # Derive dt from cumulative time if not provided.
            dt = torch.zeros_like(t)
            dt[:, 1:] = t[:, 1:] - t[:, :-1]

        h = torch.zeros(B, D, device=x.device, dtype=x_emb.dtype)

        for i in range(L):
            # Skip padded positions (item_id == 0)
            mask = (x[:, i] != 0).float().unsqueeze(-1)  # [B, 1]
            
            if i > 0:
                # ODE evolve h from t_{i-1} to t_i with dt_i.
                # Use cumulative time for integration limits and dt for conditioning.
                t_prev = t[:, i - 1].unsqueeze(-1)  # [B, 1]
                dt_i = dt[:, i].clamp(min=1e-6).unsqueeze(-1)  # [B, 1]

                if self.solver_mode == 'rk4':
                    h_evolved = self.solver.rk4_step(h, t_prev, dt_i, dt_i)
                elif self.solver_mode == 'euler':
                    h_evolved = self.solver.euler_step(h, t_prev, dt_i, dt_i)
                else:
                    h_evolved = h

                # Only evolve non-padded positions
                h = h_evolved * mask + h * (1 - mask)

            # GRU update: incorporate observation x_i
            h_new = self.gru_cell(x_emb[:, i, :], h)
            h = h_new * mask + h * (1 - mask)

        return h

    def get_item_embedding(self, item_ids):
        if self.use_embedding:
            return self.item_emb(item_ids)
        else:
            return torch.zeros(item_ids.shape[0], self.hidden_dim).to(item_ids.device)
