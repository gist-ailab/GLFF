import torch
import torch.nn.functional as F

def dilate_mask(self, mask: torch.Tensor, kernel_size: int = 3):
    """
    mask: (B,1,H,W) float [0,1]
    """
    if kernel_size <= 1:
        return mask

    pad = kernel_size // 2
    kernel = torch.ones(
        1, 1, kernel_size, kernel_size,
        device=mask.device,
        dtype=mask.dtype
    )

    # dilation via max-pooling
    dilated = F.conv2d(mask, kernel, padding=pad)
    dilated = (dilated > 0).float()
    return dilated  