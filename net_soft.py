# RTS -> soft select, cross attention q: local, k,v: global
import torch
import torch.nn as nn
import torch.nn.functional as F
from glff_attention import LesionToGlobalCrossAttention, LesionToGlobalCrossAttnLayer

# -------------------------
# 1) Lesion Token Selector (LTS)
# -------------------------
class LesionTokenSelector(nn.Module):
    """
    Soft lesion token weighting (NOT hard selection)

    Returns:
      - weighted_tokens: (B, N, D)
      - patch_weight: (B, N) in [0,1]
    """
    def __init__(self, patch_size: int = 14, tau: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.tau = tau
        self.eps = eps

    def forward(self, mask: torch.Tensor, patch_tokens: torch.Tensor):
        """
        mask: (B,1,H,W) in [0,1]
        patch_tokens: (B, N, D)
        """
        B, _, H, W = mask.shape
        B2, N, D = patch_tokens.shape
        assert B == B2

        # 1) mask → patch mask
        patch_mask_2d = F.avg_pool2d(
            mask,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # (B,1,Hp,Wp)

        patch_mask = patch_mask_2d.flatten(1)  # (B,N)

        # 2) soft weight (0~1)
        #    below tau → 0, above tau → linear scaling
        weight = (patch_mask - self.tau) / (1.0 - self.tau + self.eps)
        weight = weight.clamp(min=0.0, max=1.0)  # (B,N)

        # 3) apply weight to tokens
        weighted_tokens = patch_tokens * weight.unsqueeze(-1)  # (B,N,D)

        return weighted_tokens, weight
    
    
# -------------------------
# 3) Fusion + classifier head
# -------------------------
class FusionHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, fusion: str = "concat", hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        assert fusion in ["concat", "gate"], "fusion must be concat or gate"
        self.fusion = fusion

        if fusion == "concat":
            in_dim = embed_dim * 2
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # gate: alpha = sigmoid(W[f_g||f_l])
            in_dim = embed_dim * 2
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
        cross_attn_layers: int = 1,
        fusion: str = "concat", # concat or gate
        hidden_dim: int = 512,
        dropout: float = 0.0,
        use_global_cls: bool = False,  # False: meanpool patch tokens, True: use CLS for global vector
    ):
        super().__init__()
        self.global_encoder = global_encoder
        self.local_encoder = local_encoder
        self.use_global_cls = use_global_cls

        self.lts = LesionTokenSelector(patch_size=patch_size, tau=tau)
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
            f_global = g_cls
        else:
            f_global = g_patch.mean(dim=1)

        # -------------
        # Local branch (masked image)
        # -------------
        masked_image = image * mask  # broadcast: (B,3,H,W) * (B,1,H,W)
        l_tokens = self.local_encoder.forward_features(masked_image)  # (B, 1+N, D)
        l_patch = l_tokens[:, 1:, :]                                  # (B, N, D)

        # LTS: soft weighting
        l_weighted, patch_weight = self.lts(mask, l_patch)  # (B,N,D), (B,N)

        # Cross-attention: weighted local tokens → queries
        z = self.cross_attn(
            q_list=[l_weighted[b] for b in range(l_weighted.size(0))],
            kv_tokens=g_patch
        )

        f_local = []
        for b in range(len(z)):
            w = patch_weight[b].unsqueeze(-1)  # (N,1)
            z_b = z[b]

            pooled = (z_b * w).sum(dim=0) / (w.sum() + 1e-6)
            f_local.append(pooled)

        f_local = torch.stack(f_local, dim=0)  # (B,D)

        # -------------
        # Fusion + classifier
        # -------------
        logits = self.head(f_global, f_local)

        return logits
