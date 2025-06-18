import os
import glob
import natsort
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# —– Optimizer 생성 함수
# torch.optim.AdamW 사용 예시
def Make_Optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# —– LR Scheduler 생성 함수
# torch.optim.lr_scheduler.StepLR 사용 예시
def Make_LR_Scheduler(optimizer, step_size: int = 30, gamma: float = 0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# —– Dice Loss 정의
def DiceLoss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    N, C, H, W = pred.shape
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    if C > 1:
        target_one_hot = (
            F.one_hot(target.long(), num_classes=C)
             .permute(0,3,1,2)
             .float()
        )
    else:
        target_one_hot = target.float().unsqueeze(1)
    pred_flat = pred.contiguous().view(N, C, -1)
    targ_flat = target_one_hot.contiguous().view(N, C, -1)
    inter = (pred_flat * targ_flat).sum(-1)
    union = pred_flat.sum(-1) + targ_flat.sum(-1)
    dice_score = (2 * inter + smooth) / (union + smooth)
    return 1. - dice_score.mean()

class SegmentationLoss(nn.Module):
    def __init__(self,
                 mode: str = 'multiclass',
                 weight_ce: float = 0.5,
                 weight_dice: float = 0.5):
        super().__init__()
        assert mode in ['binary','multiclass'], "mode는 'binary' 또는 'multiclass'"
        self.mode = mode
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

        if mode == 'binary':
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target이 (N,1,H,W) 형태라면 (N,H,W)로
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        if self.mode == 'binary':
            tgt_bce = target.unsqueeze(1)  # (N,1,H,W)
            bce = self.ce_loss(logits, tgt_bce.float())

            prob = torch.sigmoid(logits)
            dice = DiceLoss(prob, target)  # DiceLoss는 (N,H,W)도 처리 가능
            return self.weight_ce * bce + self.weight_dice * dice

        else:
            # multi-class
            tgt = target.long()  # (N,H,W)
            
            # ─── CrossEntropy만 non-deterministic 허용 ───
            prev = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
            ce = self.ce_loss(logits, tgt)
            torch.use_deterministic_algorithms(prev)
            # ─────────────────────────────────────────────

            prob = F.softmax(logits, dim=1)
            dice = DiceLoss(prob, target)
            return self.weight_ce * ce + self.weight_dice * dice

def Make_Loss_Function(number_of_classes: int,
                       weight_ce: float = 0.3,
                       weight_dice: float = 0.7) -> nn.Module:
    mode = 'binary' if number_of_classes <= 2 else 'multiclass'
    return SegmentationLoss(mode=mode, weight_ce=weight_ce, weight_dice=weight_dice)