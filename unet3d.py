import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time

#vanilla unet for 3d image segmentation https://github.com/aghdamamir/3D-UNet/blob/main/unet3d.py
# exampel of 3d mri medium article: https://medium.com/@rehman.aimal/implement-3d-unet-for-cardiac-volumetric-mri-scans-in-pytorch-79f8cca7dc68  https://github.com/aimalrehman92/CardiacMRI_3D_UNet_Pytorch/blob/master/3D_Cardiac_UNet.ipynb
#3d unet paper https://arxiv.org/pdf/1606.06650v1 
#dual stream implementation attempt with chatgpt and diagram from paper: 
#   Elghazy et al. #https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12708 
# Ronneberger et al. 2015 https://arxiv.org/pdf/1505.04597#page=1.96 

"""
Dual-Stream 3D UNet Architecture based on:
"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
Adapted to process two modalities (T1 and T2) with separate decoders.
Each encoder branch is identical, and their corresponding outputs are fused
using 1x1x1 convolution. Then, two separate decoders (synthesis paths) produce
modality-specific segmentation outputs.
"""

########################################
# Basic Building Blocks (same as provided code)
########################################

class Conv3DBlock(nn.Module):
    """
    Basic block for double 3x3x3 convolutions with BatchNorm3d and ReLU.
    If bottleneck==False, a 2x2x2 max pooling is applied.
    """
    def __init__(self, in_channels, out_channels, bottleneck=False):
        super(Conv3DBlock, self).__init__()
        # Following the paper’s suggestion: use an intermediate number of channels (out_channels//2)
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = bottleneck
        if not self.bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        if self.bottleneck:
            return out, out#return same feature for skip (won't be used) and for further processing.
        else:
            return self.pooling(out), out  #return pooled output and residual (skip connection)

class UpConv3DBlock(nn.Module):
    """
    Decoder block: up-convolves with a 2x2x2 transpose convolution, concatenates with the corresponding
    encoder residual (skip) connection, and applies two 3x3x3 convolutions with BatchNorm3d and ReLU.
    If last_layer==True, a final 1x1x1 convolution produces the segmentation output.
    """
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None):
        super(UpConv3DBlock, self).__init__()
        #upsample: transpose conv with kernel 2 and stride 2.
        self.upconv1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(in_channels + res_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels // 2)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.last_layer = last_layer
        if self.last_layer:
            assert num_classes is not None, "num_classes must be provided for the last layer"
            self.conv3 = nn.Conv3d(in_channels // 2, num_classes, kernel_size=1)
            
    def forward(self, x, residual):
        x = self.upconv1(x)
        #wth same padding, the upsamped x and the residual should have matching spatial dimensions.
        if x.shape[2:] != residual.shape[2:]:
            residual = F.interpolate(residual, size=x.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, residual], dim=1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.last_layer:
            x = self.conv3(x)
        return x

########################################
#Vanilla 3D U-Net
########################################

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512):
        super(UNet3D, self).__init__()
        level1, level2, level3 = level_channels
        self.a_block1 = Conv3DBlock(in_channels, level1)
        self.a_block2 = Conv3DBlock(level1, level2)
        self.a_block3 = Conv3DBlock(level2, level3)
        self.bottleNeck = Conv3DBlock(level3, bottleneck_channel, bottleneck=True)
        self.s_block3 = UpConv3DBlock(bottleneck_channel, res_channels=level3)
        self.s_block2 = UpConv3DBlock(level3, res_channels=level2)
        self.s_block1 = UpConv3DBlock(level2, res_channels=level1, last_layer=True, num_classes=num_classes)
    
    def forward(self, x):
        #encoder path
        out, res1 = self.a_block1(x)
        out, res2 = self.a_block2(out)
        out, res3 = self.a_block3(out)
        #bottleneck
        out, _ = self.bottleNeck(out) 
        #decoder path
        out = self.s_block3(out, res3)
        out = self.s_block2(out, res2)
        out = self.s_block1(out, res1)
        return out

########################################
#DS 3D U-NET
########################################

class DualStreamUNet3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, level_channels=[64, 128, 256], bottleneck_channel=512):
        """
        Dual-stream architecture that builds two UNet3D branches for T1 and T2.
        At each corresponding level (skip connections and bottleneck), features from both streams
        are fused via concatenation followed by a 1×1 convolution. Then, each branch's decoder is run separately.
        """
        super(DualStreamUNet3D, self).__init__()
        level1, level2, level3 = level_channels
        
        #encoder path
        self.a_block1_T1 = Conv3DBlock(in_channels, level1)
        self.a_block2_T1 = Conv3DBlock(level1, level2)
        self.a_block3_T1 = Conv3DBlock(level2, level3)
        self.bottleNeck_T1 = Conv3DBlock(level3, bottleneck_channel, bottleneck=True)
        
        self.a_block1_T2 = Conv3DBlock(in_channels, level1)
        self.a_block2_T2 = Conv3DBlock(level1, level2)
        self.a_block3_T2 = Conv3DBlock(level2, level3)
        self.bottleNeck_T2 = Conv3DBlock(level3, bottleneck_channel, bottleneck=True)
        
        #fusion
        self.fuse_skip1 = nn.Conv3d(level1 * 2, level1, kernel_size=1)
        self.fuse_skip2 = nn.Conv3d(level2 * 2, level2, kernel_size=1)
        self.fuse_skip3 = nn.Conv3d(level3 * 2, level3, kernel_size=1)
        self.fuse_bottle = nn.Conv3d(bottleneck_channel * 2, bottleneck_channel, kernel_size=1)
        
        #decoder path
        #t1
        self.s_block3_T1 = UpConv3DBlock(bottleneck_channel, res_channels=level3)
        self.s_block2_T1 = UpConv3DBlock(level3, res_channels=level2)
        self.s_block1_T1 = UpConv3DBlock(level2, res_channels=level1, last_layer=True, num_classes=num_classes)
        #t2
        self.s_block3_T2 = UpConv3DBlock(bottleneck_channel, res_channels=level3)
        self.s_block2_T2 = UpConv3DBlock(level3, res_channels=level2)
        self.s_block1_T2 = UpConv3DBlock(level2, res_channels=level1, last_layer=True, num_classes=num_classes)
    
    def forward(self, x):
        """
        Expects input x of shape [B, 2, C, D, H, W] where:
         - x[:,0] is the T1 modality,
         - x[:,1] is the T2 modality.
        Returns:
         - pred_T1: segmentation prediction for T1.
         - pred_T2: segmentation prediction for T2.
        """
        #split into t1 and t2
        x_T1 = x[:, 0, ...]
        x_T2 = x[:, 1, ...]
        
        #encoder path for t1
        out1, res1_T1 = self.a_block1_T1(x_T1)
        out1, res2_T1 = self.a_block2_T1(out1)
        out1, res3_T1 = self.a_block3_T1(out1)
        out1, bottle_T1 = self.bottleNeck_T1(out1)
        
      #encoder path for t2
        out2, res1_T2 = self.a_block1_T2(x_T2)
        out2, res2_T2 = self.a_block2_T2(out2)
        out2, res3_T2 = self.a_block3_T2(out2)
        out2, bottle_T2 = self.bottleNeck_T2(out2)
        
        #skin connections + bottleneck fusion
        fuse_res1 = self.fuse_skip1(torch.cat([res1_T1, res1_T2], dim=1))
        fuse_res2 = self.fuse_skip2(torch.cat([res2_T1, res2_T2], dim=1))
        fuse_res3 = self.fuse_skip3(torch.cat([res3_T1, res3_T2], dim=1))
        fuse_bottle = self.fuse_bottle(torch.cat([bottle_T1, bottle_T2], dim=1))
        
        #decoder t1
        d3_T1 = self.s_block3_T1(fuse_bottle, fuse_res3)
        d2_T1 = self.s_block2_T1(d3_T1, fuse_res2)
        pred_T1 = self.s_block1_T1(d2_T1, fuse_res1)
        
        #decoder t2
        d3_T2 = self.s_block3_T2(fuse_bottle, fuse_res3)
        d2_T2 = self.s_block2_T2(d3_T2, fuse_res2)
        pred_T2 = self.s_block1_T2(d2_T2, fuse_res1)
        
        return pred_T1, pred_T2

########################################
#testing
########################################

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    #vanilla
    print("****** Vanilla 3D UNet (Single-Stream) ******")
    model_vanilla = UNet3D(in_channels=3, num_classes=1, level_channels=[64, 128, 256], bottleneck_channel=512).to(device)
    #according to the paper, input is 132×132×116 with 3 channels.
    summary(model_vanilla, input_size=(3, 132, 132, 116), batch_size=-1)
    
    dummy_input_vanilla = torch.randn(1, 3, 132, 132, 116).to(device)
    pred_vanilla = model_vanilla(dummy_input_vanilla)
    print("Vanilla UNet Output Shape:", pred_vanilla.shape)  # Expected: [1, 1, output_D, output_H, output_W]
    
    # # Test the dual-stream UNet with separate decoders.
    # print("****** Dual-Stream UNet with Separate Decoders ******")
    # # Input for dual-stream: Batch size 1, 2 modalities, each with 3 channels, 132x132x116.
    # model_dual = DualStreamUNet3D(in_channels=3, num_classes=1, level_channels=[64,128,256], bottleneck_channel=512).to(device)
    # summary(model_dual, input_size=(2, 3, 132, 132, 116), batch_size=1, device="cpu")
    
    # dummy_input_dual = torch.randn(1, 2, 3, 132, 132, 116).to(device)
    # pred_T1, pred_T2 = model_dual(dummy_input_dual)
    # print("Dual-Stream T1 Output Shape:", pred_T1.shape)
    # print("Dual-Stream T2 Output Shape:", pred_T2.shape)
