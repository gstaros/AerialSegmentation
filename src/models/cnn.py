import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.sigmoid(self.conv3(x))
    x = torch.round(x)

    return x