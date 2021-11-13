# ORIGNALLY FROM TIMM LIBRARY @ https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py

import torch
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

def soft_target_loss(x, y, weight = None):
    '''
    Soft target loss from timm library
    
    Parameters:
        x (torch.tensor): Outputs from model 
        y (torch.tensor): Soft class labels
        weight (torch.tensor): Weights for class balanced loss. Lenght is number of images in batchs
    '''
    loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
    if weight is not None:
        loss = weight * loss
    
    return loss.mean()

def LabelSmoothingLoss(x, target, smoothing = 0.1, weight = None):
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
    if weight is not None:
        loss = weight * loss
    
    return loss.mean()