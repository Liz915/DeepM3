import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM'18).
    Manual implementation to avoid PyTorch nn.Transformer API issues on MPS.
    """

    def __init__(self, n_items, hidden_dim, max_len=20, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.n_heads = n_heads

        self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        nn.init.xavier_normal_(self.item_emb.weight)

        self.layers = nn.ModuleList([
            SASBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, L] item IDs. Returns: [B, D] user embedding."""
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        h = self.item_emb(x) * math.sqrt(self.hidden_dim) + self.pos_emb(positions)
        h = self.dropout(h)

        # Causal mask: True = attend, False = mask out
        causal = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        # Padding mask: True = valid token
        pad_mask = (x != 0)  # [B, L]

        for layer in self.layers:
            h = layer(h, causal, pad_mask)

        h = self.final_norm(h)
        return h[:, -1, :]

    def get_item_embedding(self, item_ids):
        return self.item_emb(item_ids)


class SASBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, causal_mask, pad_mask):
        x = x + self.attn(self.norm1(x), causal_mask, pad_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, pad_mask):
        """
        x: [B, L, D]
        causal_mask: [L, L] bool, True = attend
        pad_mask: [B, L] bool, True = valid
        """
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, L, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # Apply causal mask
        attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask: don't attend to padding positions
        key_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        attn = attn.masked_fill(~key_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked rows
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class TiSASRec(nn.Module):
    """Time-Interval Aware Self-Attention (Li et al., WSDM'20).
    Extends SASRec with time interval position bias.
    """

    def __init__(self, n_items, hidden_dim, max_len=20, n_heads=2, n_layers=2,
                 dropout=0.1, time_span=256):
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.n_heads = n_heads
        self.time_span = time_span

        self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        nn.init.xavier_normal_(self.item_emb.weight)

        # Time interval bias embeddings  
        head_dim = hidden_dim // n_heads
        self.time_bias_emb = nn.Embedding(time_span + 1, n_heads)

        self.layers = nn.ModuleList([
            TiSASBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def _compute_time_bias(self, t):
        """Compute relative time interval bias for attention."""
        # t: [B, L]
        t_i = t.unsqueeze(-1)   # [B, L, 1]
        t_j = t.unsqueeze(-2)   # [B, 1, L]
        time_diff = torch.abs(t_i - t_j)  # [B, L, L]
        # Bucketize with log-scale
        time_idx = torch.log1p(time_diff).long().clamp(0, self.time_span)  # [B, L, L]
        bias = self.time_bias_emb(time_idx)  # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2)  # [B, H, L, L]
        return bias

    def forward(self, x, t):
        """
        x: [B, L] item IDs
        t: [B, L] timestamps
        Returns: [B, D] user embedding
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        h = self.item_emb(x) * math.sqrt(self.hidden_dim) + self.pos_emb(positions)
        h = self.dropout_layer(h)

        causal = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        pad_mask = (x != 0)

        time_bias = self._compute_time_bias(t)  # [B, H, L, L]

        for layer in self.layers:
            h = layer(h, causal, pad_mask, time_bias)

        h = self.final_norm(h)
        return h[:, -1, :]

    def get_item_embedding(self, item_ids):
        return self.item_emb(item_ids)


class TiSASBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.attn = TimeSelfAttention(hidden_dim, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, causal_mask, pad_mask, time_bias):
        x = x + self.attn(self.norm1(x), causal_mask, pad_mask, time_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class TimeSelfAttention(nn.Module):
    """Self-attention with additive time interval bias."""
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, pad_mask, time_bias):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # Add time interval bias
        attn = attn + time_bias

        attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        key_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(~key_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)
