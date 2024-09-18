from torch.utils.data import Dataset
from skimage import io
import torch
import numpy as np 
import pandas as pd
import torchvision.transforms as T


class ImageSegmentationDataset(Dataset):
  def __init__(self, dir_file: str, n_channels: int=5, n_classes: int=13, transform=None):
    self.image_mask_mapping = pd.read_csv(dir_file)
    self.images = np.array(self.image_mask_mapping["IMG"])
    self.masks = np.array(self.image_mask_mapping["MSK"])

    self.n_channels = n_channels
    self.n_classes = n_classes
    self.transform = transform


  def read_tiff_image(self, file_name: str) -> torch.Tensor:
    image = io.imread(file_name)
    #io.imread gives array (H, W, C)
    return torch.as_tensor(image).permute(2, 0, 1).type(torch.FloatTensor)


  def read_tiff_mask(self, mask_name: str) -> torch.Tensor:
    mask = io.imread(mask_name)

    # set all mask values from 0-12.
    mask[mask > 13] = 13
    mask = mask - 1
    return torch.from_numpy(mask).type(torch.LongTensor)


  def __len__(self):
    return len(self.images)


  def __getitem__(self, idx):
    image_filename = self.images[idx]
    mask_filename = self.masks[idx]

    image = self.read_tiff_image(image_filename)[:self.n_channels]
    mask = self.read_tiff_mask(mask_filename)

    if self.transform:
      image = self.transform(image)

      mask = mask.unsqueeze(dim=0)
      mask = self.transform(mask).squeeze(dim=0)

    return image, mask