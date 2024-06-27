from .unet_parts import Convolution, Down, Up
import torch.nn as nn

class UNet(nn.Module):
  def __init__(self, n_channels):
    super(UNet, self).__init__()
    self.in_conv = Convolution(n_channels, 64)
    self.down_conv1 = Down(64, 128)
    self.down_conv2 = Down(128, 256)
    self.down_conv3 = Down(256, 512)
    self.down_conv4 = Down(512, 1024)
    self.up1 = Up(1024, 512)
    self.up2 = Up(512, 256)
    self.up3 = Up(256, 128)
    self.up4 = Up(128, 64)
    self.out_conv = Convolution(64, 1)

  def forward(self, x):
    conv1 = self.in_conv(x)
    conv2 = self.down_conv1(conv1)
    conv3 = self.down_conv2(conv2)
    conv4 = self.down_conv3(conv3)
    conv5 = self.down_conv4(conv4)

    up_conv4 = self.up1(conv5, conv4)
    up_conv3 = self.up2(up_conv4, conv3)
    up_conv2 = self.up3(up_conv3, conv2)
    up_conv1 = self.up4(up_conv2, conv1)
    out_conv = self.out_conv(up_conv1)

    return out_conv
