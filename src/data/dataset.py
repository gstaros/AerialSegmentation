from torch.utils.data import Dataset
from torch import FloatTensor
from PIL import Image
import torchvision.transforms as T
import os



class ImageSegmentationDataset(Dataset):
  def __init__(self, 
               image_folder: str, 
               mask_folder: str, 
               split: str = "train", 
               transform = None):

    self.image_folder = image_folder
    self.mask_folder = mask_folder
    self.split = split
    self.tranform = transform


    if split == "train":
      self.mask_files = sorted(os.listdir(self.mask_folder))[:400]
      self.image_files = sorted(os.listdir(self.image_folder))[:400]

    elif split == "val":
      self.mask_files = sorted(os.listdir(self.mask_folder))[400:500]
      self.image_files = sorted(os.listdir(self.image_folder))[400:500]

    elif split == "test":
      self.mask_files = sorted(os.listdir(self.mask_folder))[500:]
      self.image_files = sorted(os.listdir(self.image_folder))[500:]


  def __len__(self):
    return len(self.image_files)


  def __getitem__(self, idx):
    image_filename = self.image_files[idx]
    mask_filename = self.mask_files[idx]
    image_path = os.path.join(self.image_folder, image_filename)
    mask_path = os.path.join(self.mask_folder, mask_filename)

    pil_to_tensor = T.Compose([T.PILToTensor(), T.Resize((256, 256))])
    image = pil_to_tensor(Image.open(image_path))
    mask = pil_to_tensor(Image.open(mask_path))

    return image.type(FloatTensor), mask.type(FloatTensor)