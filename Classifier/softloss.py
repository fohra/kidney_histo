# ORIGNALLY FROM TIMM LIBRARY @ https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py

import torch
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

def soft_target_loss(x, y):
    '''
    Soft target loss from timm library
    
    Parameters:
        x (torch.tensor): Outputs from model 
        y (torch.tensor): Soft class labels
        weight (
    '''
    loss = SoftTargetCrossEntropy()(x,y)
    
    return loss

def LabelSmoothingLoss(x, target, smoothing = 0.1):
    ''' 
    Label smoothing loss from timm library
    edited by jopo666 
    '''
    if len(target.shape) > 1:
        target = target.max(1)[1]
    logprobs = F.log_softmax(x, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = (1-smoothing) * nll_loss + smoothing * smooth_loss
    return loss.mean()