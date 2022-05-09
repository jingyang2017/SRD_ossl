from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, class_weights=None):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if class_weights is not None:
            _, preds_t = torch.max(y_t, 1)
            weights = class_weights[preds_t]
            weights = weights.unsqueeze(1)
            l_kl = F.kl_div(p_s, p_t, reduction='none')
            loss = torch.sum(l_kl * weights) * (self.T**2) / y_s.shape[0]
        else:
            loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss
