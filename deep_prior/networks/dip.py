import torch
import torch.nn as nn
from networks.model_utils import *
from networks.sngan.snlayers.snconv2d import SNConv2d

z_dim = 64

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        
    def forward(self, input):
        return torch.reshape(input, (input.size(0),*self.target_shape))
        
class DecoderDip(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
            
            UnFlatten((ndf*16, 1, 1)),
            # state size. (ndf*16) x 1 x 1
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf*16, ndf*8, 2, 1, 1),
            nn.BatchNorm2d(ndf*8),
            nn.SELU(),

            
            nn.Conv2d(ndf*8, ndf*8, 3, 1, 1),
            nn.BatchNorm2d(ndf*8),
            nn.SELU(),
            
            # state size. (ndf*8) x 3 x 3
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf*8, ndf*4, 3, 1, 1),
            nn.BatchNorm2d(ndf*4),
            nn.SELU(),

            nn.Conv2d(ndf*4, ndf*4, 3, 1, 1),
            nn.BatchNorm2d(ndf*4),
            nn.SELU(),

            # state size. (ndf*4) x 6 x 6
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf*4, ndf*2, 3, 1, 1),
            nn.BatchNorm2d(ndf*2),
            nn.SELU(),

            nn.Conv2d(ndf*2, ndf*2, 3, 1, 1),
            nn.BatchNorm2d(ndf*2),
            nn.SELU(),

            # state size (ndf*2) x 12 x 12
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf*2, ndf*1, 3, 1, 2),
            nn.BatchNorm2d(ndf*1),
            nn.SELU(),
            
            nn.Conv2d(ndf, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.SELU(),

            # state size (ndf) x 26, 26
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf*1, 2, 3, 1, 1),
            nn.BatchNorm2d(2),
            nn.SELU(),

            nn.Conv2d(2, 2, 3, 1, 1),
            nn.BatchNorm2d(2),
            nn.SELU(),

            # state size 2 x 54 x 54
            nn.Conv2d(2,1, 4, 1, 1),
            nn.Sigmoid()
            # output 1 x 51 x 51            
        )

    def forward(self, input):
        # for layer in self.main:
        #     input = layer(input)
        #     print(input.shape)
        # return input
        return self.main(input)
