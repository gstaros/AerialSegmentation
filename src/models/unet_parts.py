import torch.nn as nn
import torch



class Convolution(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None):
      super().__init__()
      if not mid_channels:
          mid_channels = out_channels
      self.double_conv = nn.Sequential(
          nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(mid_channels),
          nn.ReLU(inplace=True),

          nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
      )

  def forward(self, x):
      return self.double_conv(x)



class Upscale(nn.Module):
  def __init__(self, in_channels, scale_factor=2):
    super(Upscale, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, in_channels // scale_factor, kernel_size=scale_factor, stride=scale_factor)

  def forward(self, x):
    return self.up(x)



class Down(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Down, self).__init__()
    self.maxpool = nn.MaxPool2d(2)
    self.conv = Convolution(in_channels, out_channels)

  def forward(self, x):
    return self.conv(self.maxpool(x))


class ConcatenateToConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConcatenateToConv, self).__init__()
    self.conv = Convolution(in_channels, out_channels)

  def forward(self, x):
    x = torch.cat([*x], dim=1)
    return self.conv(x)


class Up(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Up, self).__init__()
    self.up = Upscale(in_channels)
    self.conv = ConcatenateToConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    x = self.conv((x1, x2))
    return x

