import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recover, gt):
        loss_pose = F.l1_loss(recover, gt, reduction='mean')
        return loss_pose
