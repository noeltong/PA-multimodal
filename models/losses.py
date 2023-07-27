from torch import nn
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

class MixedLoss(nn.Module):
    def __init__(self, alpha=0.25) -> None:
        super().__init__()
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(kernel_size=3, betas=(0.0448, 0.2856, 0.3001))
        self.l1 = CharbonnierLoss(eps=1e-5)
        self.alpha = alpha

    def forward(self, x, y):
        return self.alpha * (1 - self.ms_ssim(x, y)) + (1 - self.alpha) * self.l1(x, y)
    

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
