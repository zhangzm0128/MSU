import os
from turtle import forward
import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        # self.loss = getattr(self, loss_name)

    def MSE(self):
        # return nn.MSELoss(reduce=True, reduction='mean')
        return nn.MSELoss(reduce=False)
    
    def RLoss(self):
        return RLoss()

    def R_MSELoss(self):
        return R_MSELoss()

    def MaskLoss(self, active_loss=None):
        return MaskLoss(active_loss)
        
    def VarMseLoss(self):
        return VarMseLoss()

class MaskLoss(nn.Module):
    def __init__(self, active_loss=None):
        super(MaskLoss, self).__init__()
        self.active_loss = active_loss
            
        self.bce = nn.BCELoss()

    def cos_loss(self, y_pred, y_true):
        return 1 - F.cosine_similarity(y_pred, y_true).abs().mean()


    def activation_loss(self, mask):
        activate_point = torch.sum(mask, dim=1)
        # return torch.pow(torch.div(activate_point, mask.shape[1]), 2).mean()
        # return torch.pow(torch.div(activate_point - 10, mask.shape[1]), 2).mean()
        return ((activate_point-1) * torch.log10(activate_point) / 250).mean()
    
    def activation_loss_test(self, mask):
        activate_point = torch.sum(mask, dim=1)
        return torch.sqrt(torch.pow(torch.div(activate_point-10, mask.shape[1]), 2)).mean()

    def forward(self, mask, mask_before=None):
        # mask shape (batch_size, dim)
        if self.active_loss is None:
            loss_activate = self.activation_loss(mask)
        else:
            loss_activate = self.activation_loss_test(mask) 

        if mask_before is not None:
            mask_before_calc = torch.repeat_interleave(mask_before.unsqueeze(dim=0), repeats=mask.shape[0], dim=0)
            # loss_sim = self.cos_loss(mask, mask_before_calc)
            loss_sim = self.bce(mask, mask_before_calc)
            return loss_activate, loss_sim

        return loss_activate

class RLoss(nn.Module):
    def __init__(self):
        super(RLoss, self).__init__()
    def forward(self, y_pred, y_true):

        y_pred_ = torch.flatten(y_pred, start_dim=1)
        y_true_ = torch.flatten(y_true, start_dim=1)

        y_pred_mean = torch.mean(y_pred_, axis=1).view(-1, 1)
        y_true_mean = torch.mean(y_true_, axis=1).view(-1, 1)

        R = torch.sum((y_pred_ - y_pred_mean) * (y_true_ - y_true_mean), axis=1) / \
            (torch.sqrt(torch.sum((y_pred_ - y_pred_mean) ** 2, axis=1)) * torch.sqrt(torch.sum((y_true_ - y_true_mean) ** 2, axis=1)))

        return 1 - R**2

class R_MSELoss(nn.Module):
    def __init__(self):
        super(R_MSELoss, self).__init__()
        self.mse = nn.MSELoss(reduce=False)
        self.r = RLoss()
    def forward(self, y_pred, y_true):
        mse_loss = torch.flatten(self.mse(y_pred, y_true), start_dim=1).mean(axis=1)
        r_loss = self.r(y_pred, y_true)
        return mse_loss + r_loss
        
class VarMseLoss(nn.Module):
    def __init__(self):
        super(VarMseLoss, self).__init__()
        self.mse = nn.MSELoss(reduce=False)
    def forward(self, y_pred, y_true):
        mse_loss = torch.flatten(self.mse(y_pred, y_true), start_dim=1).mean(axis=1)
        
        # var loss over temporal
        var_pred = torch.var(y_pred, 1)
        var_true = torch.var(y_true, 1)
        var_loss = ((var_pred - var_true) ** 2).mean(axis=1)
        
        return mse_loss + var_loss
    
    
    
    
    
    
