#https://github.com/JunZengz/RMAMamba 

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

""" Loss Functions -------------------------------------- """
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
    
#CHATGPT VERSION OF BCEDiceloss
    # class BCEDiceLoss(nn.Module):
    # def __init__(self, weight=None):
    #     super(BCEDiceLoss, self).__init__()
    #     self.weight = weight

    # def forward(self, inputs, targets):
    #     # Compute Binary Cross Entropy loss with logits.
    #     bce = F.binary_cross_entropy_with_logits(inputs, targets, weight=self.weight)
        
    #     # Compute Dice loss.
    #     inputs = torch.sigmoid(inputs)
    #     smooth = 1.0
    #     # Flatten the predictions and targets
    #     inputs_flat = inputs.view(inputs.size(0), -1)
    #     targets_flat = targets.view(targets.size(0), -1)
    #     intersection = (inputs_flat * targets_flat).sum(1)
    #     dice = (2. * intersection + smooth) / (inputs_flat.sum(1) + targets_flat.sum(1) + smooth)
    #     dice_loss = 1 - dice.mean()
        
    #     return bce + dice_loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        
        #flatten predictions and targets
        inputs = inputs.view(-1) 
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

## https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/319452
def hd_dist(preds, targets):
    haussdorf_dist = directed_hausdorff(preds, targets)[0]
    return haussdorf_dist