from .reins import Reins
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
 

class ReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins = Reins(
            num_layers = kwargs['depth'],
            embed_dims = kwargs['embed_dim'],
            patch_size = kwargs['patch_size'],
        )

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x
    


    def forward_features_full_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.reins.return_auto(outs)

    def forward_features_no_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "linear"])
        set_train(self, ["reins", "linear"])



        
class ReinsDinoVisionTransformer_3_head(DinoVisionTransformer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins1 = Reins(
            num_layers = kwargs['depth'],
            embed_dims = kwargs['embed_dim'],
            patch_size = kwargs['patch_size'],
        )

        self.reins2 = Reins(
            num_layers = kwargs['depth'],
            embed_dims = kwargs['embed_dim'],
            patch_size = kwargs['patch_size'],
        )
        
        self.reins3 = Reins(
            num_layers = kwargs['depth'],
            embed_dims = kwargs['embed_dim'],
            patch_size = kwargs['patch_size'],
        )
        


    def forward_features1(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins1.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x
    
    def forward_features2(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x
    
    def forward_features3(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x

    def forward_features_no_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        return x
    
    def forward(self, x, mode='rein1', masks=None):
        # 편의용 통합 forward
        if mode == 'rein1':
            f = self.forward_features1(x, masks)
            return self.linear1(f[:, 0, :])
        elif mode == 'rein2':
            f = self.forward_features2(x, masks)
            return self.linear2(f[:, 0, :])
        elif mode == 'rein3':
            f = self.forward_features3(x, masks)
            return self.linear3(f[:, 0, :])
        elif mode == 'no_rein':
            f = self.forward_features_no_rein(x, masks)
            return self.linear(f[:, 0, :])
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            set_requires_grad(self, ['reins1', 'reins2', 'reins3', 'linear1', 'linear2', 'linear3'])
            set_train(self, ['reins1', 'reins2', 'reins3', 'linear1', 'linear2', 'linear3'])
        return self
        
        
                