import torch
import torch.nn as nn

class GRUBaseline(nn.Module):
    def __init__(self, n_items, hidden_dim):
        super().__init__()
        # KBS Embedding 
        self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)
        nn.init.xavier_normal_(self.item_emb.weight)

        #  nn.GRU (Fused Kernel) GRUCell
        #  MPS/CUDA  for-loop  10-100 
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        """
        x: [Batch, Seq_Len] (Item IDs)
        Return: [Batch, Hidden_Dim] (User Final Embedding)
        """
        emb = self.item_emb(x)          # [B, L, D]
        
        # nn.GRU  Python Loop
        # out: [B, L, D], h_n: [1, B, D]
        _, h_n = self.gru(emb)
        
        #  User Embedding
        return h_n.squeeze(0) # [B, D]

    def get_item_embedding(self, ids):
        return self.item_emb(ids)