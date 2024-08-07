import torch.nn as nn

class ViT(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(ViT, self).__init__()
    self.keys = keys
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
