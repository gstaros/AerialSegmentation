from .unet_parts import Convolution, Down, Up, ConcatenateToConv, Upscale
import torch.nn as nn



class Convolution_better(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(UNet, self).__init__()
    self.in_conv = Convolution(n_channels, 64)
    self.conv_1_0 = Down(64, 128)
    self.conv_2_0 = Down(128, 256)
    self.conv_3_0 = Down(256, 512)
    self.conv_4_0 = Down(512, 1024)
    self.up_1 = Up(1024, 512)
    self.up_2 = Up(512, 256)
    self.up_3 = Up(256, 128)
    self.up_4 = Up(128, 64)
    self.out_conv = Convolution(64, n_classes)

  def forward(self, x):
    x_0_0 = self.in_conv(x)
    x_1_0 = self.conv_1_0(x_0_0)
    x_2_0 = self.conv_2_0(x_1_0)
    x_3_0 = self.conv_3_0(x_2_0)
    x_4_0 = self.conv_4_0(x_3_0)

    x_4_1 = self.up_1(x_4_0, x_3_0)
    x_3_1 = self.up_2(x_4_1, x_2_0)
    x_2_1 = self.up_3(x_3_1, x_1_0)
    x_1_1 = self.up_4(x_2_1, x_0_0)
    out_conv = self.out_conv(x_1_1)

    return out_conv
  


class FlexibleUNet(nn.Module):
  def __init__(self, n_channels, n_classes, n_layers=4):
    super(FlexibleUNet, self).__init__()
    self.n_layers = n_layers
    layer_out = 64
    self.in_conv = Convolution(n_channels, layer_out)

    #Create Down Layers objects
    down_conv_list = []
    for _ in range(n_layers):
       down_conv_list.append(Down(layer_out, layer_out*2))
       layer_out = layer_out*2
    self.down_conv = nn.Sequential(*down_conv_list)

    #Create Up Layers objects
    up_conv_list = []
    for _ in range(n_layers):
       up_conv_list.append(Up(layer_out, layer_out//2))
       layer_out = layer_out//2
    self.up_conv = nn.Sequential(*up_conv_list)

    self.out_conv = Convolution(layer_out, n_classes)


  def forward(self, x):
    #In Conv
    conv_in = self.in_conv(x)
    last_layer = conv_in

    #Down Layers 
    down_conv_output = []
    for idx, down_layer in enumerate(self.down_conv):
      down_x = down_layer(last_layer)
      down_conv_output.append(down_x)
      last_layer = down_conv_output[idx]

    #Up Layers
    up_conv_output = []
    for idx, up_layer in enumerate(self.up_conv):
      if -idx-2 < -self.n_layers:
        up_x = up_layer(last_layer, conv_in)
        up_conv_output.append(up_x)
      else:
        up_x = up_layer(last_layer, down_conv_output[-idx-2])
        up_conv_output.append(up_x)
      last_layer = up_conv_output[idx]
  
    #Out Conv
    out_conv = self.out_conv(last_layer)

    return out_conv



class UNetPlusPlus(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(UNetPlusPlus, self).__init__()

    layers_dim = [64, 128, 256, 512, 1024]

    self.in_conv = Convolution(n_channels, 64)
    self.conv_1_0 = Down(layers_dim[0], layers_dim[1])
    self.conv_2_0 = Down(layers_dim[1], layers_dim[2])
    self.conv_3_0 = Down(layers_dim[2], layers_dim[3])
    self.conv_4_0 = Down(layers_dim[3], layers_dim[4])

    self.up_1 = Upscale(in_channels=layers_dim[1], scale_factor=2)
    self.up_2 = Upscale(in_channels=layers_dim[2], scale_factor=2)
    self.up_3 = Upscale(in_channels=layers_dim[3], scale_factor=2)
    self.up_4 = Upscale(in_channels=layers_dim[4], scale_factor=2)

    self.conv_x_0_1 = ConcatenateToConv(layers_dim[0]*2, layers_dim[0])
    self.conv_x_1_1 = ConcatenateToConv(layers_dim[1]*2, layers_dim[1])
    self.conv_x_2_1 = ConcatenateToConv(layers_dim[2]*2, layers_dim[2])
    self.conv_x_3_1 = ConcatenateToConv(layers_dim[3]*2, layers_dim[3])

    self.conv_x_0_2 = ConcatenateToConv(layers_dim[0]*3, layers_dim[0])
    self.conv_x_1_2 = ConcatenateToConv(layers_dim[1]*3, layers_dim[1])
    self.conv_x_2_2 = ConcatenateToConv(layers_dim[2]*3, layers_dim[2])

    self.conv_x_0_3 = ConcatenateToConv(layers_dim[0]*4, layers_dim[0])
    self.conv_x_1_3 = ConcatenateToConv(layers_dim[1]*4, layers_dim[1])

    self.conv_x_0_4 = ConcatenateToConv(layers_dim[0]*5, layers_dim[0])

    self.out_conv = Convolution(layers_dim[0], n_classes)

  def forward(self, x):
    x_0_0 = self.in_conv(x)
    x_1_0 = self.conv_1_0(x_0_0)
    x_2_0 = self.conv_2_0(x_1_0)
    x_3_0 = self.conv_3_0(x_2_0)
    x_4_0 = self.conv_4_0(x_3_0)

    x_0_1 = self.conv_x_0_1((self.up_1(x_1_0), x_0_0))
    x_1_1 = self.conv_x_1_1((self.up_2(x_2_0), x_1_0))
    x_2_1 = self.conv_x_2_1((self.up_3(x_3_0), x_2_0))
    x_3_1 = self.conv_x_3_1((self.up_4(x_4_0), x_3_0))

    x_0_2 = self.conv_x_0_2((self.up_1(x_1_1), x_0_0, x_0_1))
    x_1_2 = self.conv_x_1_2((self.up_2(x_2_1), x_1_0, x_1_1))
    x_2_2 = self.conv_x_2_2((self.up_3(x_3_1), x_2_0, x_2_1))

    x_0_3 = self.conv_x_0_3((self.up_1(x_1_2), x_0_0, x_0_1, x_0_2))
    x_1_3 = self.conv_x_1_3((self.up_2(x_2_2), x_1_0, x_1_1, x_1_2))

    x_0_4 = self.conv_x_0_4((self.up_1(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3))

    out_conv = self.out_conv(x_0_4)

    return out_conv