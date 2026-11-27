import torch
import torch.nn as nn


class LesionToGlobalCrossAttnLayer(nn.Module):
    """
    Single cross-attention layer:
      Q: lesion tokens
      K,V: global tokens
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        """
        q : (1, N_l, D)
        kv: (1, N, D)
        """
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        z, _ = self.attn(qn, kvn, kvn, need_weights=False)
        z = self.proj_drop(z)
        return q + z   # residual


class LesionToGlobalCrossAttention(nn.Module):
    """
    Stack of cross-attention layers
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            LesionToGlobalCrossAttnLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, q_list, kv_tokens: torch.Tensor):
        """
        q_list   : list length B, each (N_l_i, D)
        kv_tokens: (B, N, D)

        returns:
          z_list  : list length B, each (N_l_i, D)
        """
        B, _, _ = kv_tokens.shape
        z_list = []

        for b in range(B):
            q = q_list[b].unsqueeze(0)          # (1, N_l, D)
            kv = kv_tokens[b:b+1]              # (1, N, D)

            for layer in self.layers:
                q = layer(q, kv)

            z_list.append(q.squeeze(0))         # (N_l, D)

        return z_list
