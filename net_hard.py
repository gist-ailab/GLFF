# RTS -> hard select, cross attention q: local, k,v: global
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1) Lesion Token Selector (LTS)
# -------------------------
class LesionTokenSelector(nn.Module):
    """
    Pixel mask (B,1,H,W) -> PatchMask (B, N) via avgpool
    then selects tokens where PatchMask > tau (hard selection).
    Optionally enforces min_tokens via top-k fallback.

    Returns:
      - selected_tokens: list[Tensor], len=B, each is (N_l_i, D)
      - sel_mask: BoolTensor (B, N) selection mask over patches
      - patch_mask: FloatTensor (B, N) in [0,1]
    """
    def __init__(self, patch_size: int = 14, tau: float = 0.5, min_tokens: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.tau = tau
        self.min_tokens = min_tokens

    @torch.no_grad()
    def forward(self, mask: torch.Tensor, patch_tokens: torch.Tensor):
        """
        mask: (B,1,H,W) float in [0,1] (or 0/255 already scaled to 0~1)
        patch_tokens: (B, N, D)  (CLS removed already)
        """
        assert mask.dim() == 4 and mask.size(1) == 1, "mask must be (B,1,H,W)"
        B, _, H, W = mask.shape
        B2, N, D = patch_tokens.shape
        assert B == B2, "batch mismatch"

        # avg pool to patch grid
        # assume H,W divisible by patch_size (e.g., 224 and 14)
        ps = self.patch_size
        patch_mask_2d = F.avg_pool2d(mask, kernel_size=ps, stride=ps)  # (B,1,Hp,Wp)
        patch_mask = patch_mask_2d.flatten(1)  # (B, N)

        # hard selection
        sel = patch_mask > self.tau  # (B, N) bool

        # ensure at least min_tokens selected per sample (fallback to top-k)
        if self.min_tokens is not None and self.min_tokens > 0:
            for b in range(B):
                if sel[b].sum().item() < self.min_tokens:
                    # pick top-k by patch_mask
                    k = min(self.min_tokens, N)
                    topk_idx = torch.topk(patch_mask[b], k=k, largest=True).indices
                    sel[b].zero_()
                    sel[b, topk_idx] = True

        selected_tokens = [patch_tokens[b, sel[b]] for b in range(B)]  # each (N_l_i, D)
        return selected_tokens, sel, patch_mask


# -------------------------
# 2) Cross-Attention block (lesion queries attend to global tokens)
# -------------------------
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



# -------------------------
# 3) Fusion + classifier head
# -------------------------
class FusionHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, fusion: str = "concat", hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        assert fusion in ["concat", "gate"], "fusion must be concat or gate"
        self.fusion = fusion

        if fusion == "concat":
            # in_dim = embed_dim * 2
            in_dim = embed_dim * 3   # CLS + patch_mean + local
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # gate: alpha = sigmoid(W[f_g||f_l])
            # in_dim = embed_dim * 2
            in_dim = embed_dim * 3   # CLS + patch_mean + local
            self.gate = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
                nn.Sigmoid()
            )
            self.cls = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, f_global: torch.Tensor, f_local: torch.Tensor):
        """
        f_global: (B,D)
        f_local : (B,D)
        """
        if self.fusion == "concat":
            f = torch.cat([f_global, f_local], dim=-1)
            return self.mlp(f)
        else:
            gate_in = torch.cat([f_global, f_local], dim=-1)
            alpha = self.gate(gate_in)  # (B,1)
            f = alpha * f_global + (1.0 - alpha) * f_local
            return self.cls(f)


# -------------------------
# 4) Full model: Global DINOv2 + Local DINOv2 + LTS + CrossAttn + Fusion
# -------------------------
class GlobalLocalLesionClassifier(nn.Module):
    """
    Inputs:
      image: (B,3,224,224)
      mask : (B,1,224,224) in [0,1]
    Outputs:
      logits: (B,num_classes)
    """
    def __init__(
        self,
        global_encoder,          # ReinsDinoVisionTransformer
        local_encoder,           # ReinsDinoVisionTransformer (can share weights or separate instance)
        embed_dim: int,
        num_classes: int,
        patch_size: int = 14,
        tau: float = 0.5,
        min_tokens: int = 4,
        num_heads: int = 8,
        cross_attn_layers: int = 2, 
        fusion: str = "concat",
        hidden_dim: int = 512,
        dropout: float = 0.0,
        use_global_cls: bool = False,  # False: meanpool patch tokens, True: use CLS for global vector
    ):
        super().__init__()
        self.global_encoder = global_encoder
        self.local_encoder = local_encoder
        self.use_global_cls = use_global_cls

        self.lts = LesionTokenSelector(patch_size=patch_size, tau=tau, min_tokens=min_tokens)
        self.cross_attn = LesionToGlobalCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.head = FusionHead(embed_dim=embed_dim, num_classes=num_classes, fusion=fusion,
                               hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        """
        image: (B,3,224,224)
        mask : (B,1,224,224) float in [0,1]
        """
        # -------------
        # Global branch
        # -------------
        g_tokens = self.global_encoder.forward_features(image)  # (B, 1+N, D)
        g_cls = g_tokens[:, 0, :]                               # (B, D)
        g_patch = g_tokens[:, 1:, :]                            # (B, N, D)


        if self.use_global_cls:
            # f_global = g_cls
            f_global = torch.cat([g_cls, g_patch.mean(dim=1)], dim=-1)
        else:
            f_global = g_patch.mean(dim=1)

        # -------------
        # Local branch (masked image)
        # -------------
        masked_image = image * mask  # broadcast: (B,3,H,W) * (B,1,H,W)
        l_tokens = self.local_encoder.forward_features(masked_image)  # (B, 1+N, D)
        l_patch = l_tokens[:, 1:, :]                                  # (B, N, D)

        # -------------
        # LTS: select lesion patch tokens from local tokens using patch-wise mask
        # -------------
        q_list, sel_mask, patch_mask = self.lts(mask, l_patch)  # list[(N_l_i,D)], (B,N), (B,N)

        # -------------
        # Cross-attention: lesion queries attend to all global patch tokens (K,V)
        # -------------
        z_list = self.cross_attn(q_list, g_patch)  # list[(N_l_i,D)]

        # -------------
        # Pool lesion tokens -> fixed vector f_local (B,D)
        # -------------
        f_local = []
        for z in z_list:
            # mean pooling over lesion tokens
            f_local.append(z.mean(dim=0))
        f_local = torch.stack(f_local, dim=0)  # (B,D)

        # -------------
        # Fusion + classifier
        # -------------
        logits = self.head(f_global, f_local)

        return logits
