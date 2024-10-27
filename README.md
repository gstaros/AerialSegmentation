# AerialSegmentation

This project aims to design and evaluate the effectiveness of various neural network models in generating accurate segmentation masks from aerial images using the FLAIR-1 dataset. This dataset comprises **77,412 images** covering various regions in France. The project investigates the performance of four segmentation models, including **UNet, UNet++, ViTSegmenter, and ResViTSegmenter**, to determine which models produce the most accurate and effective segmentation masks.

## Project Overview

The main objectives of this project were:
1. To prepare, train, and evaluate four different neural network models for segmentation.
2. To assess the models using key metrics like accuracy and mean Intersection over Union (mIoU).
3. To provide a GitHub repository enabling users to leverage these models and their pre-trained weights for aerial image segmentation. Additionally, users can train the models on their custom datasets, tailoring them for specific applications.

## Models Used

1. **UNet**: A classical segmentation model with high mIoU but moderate accuracy.
2. **UNet++**: Enhanced version of UNet, achieving the highest accuracy among the models tested.
3. **ViTSegmenter**: Vision Transformer-based model with high mIoU but relatively lower accuracy. Based on this [paper](https://arxiv.org/abs/2105.05633) 
4. **ResViTSegmenter**: Combines ResNet and Vision Transformer components, achieving high accuracy and strong mIoU values.

## Results

Each model was evaluated based on:
- **Accuracy**: Measures the overall correctness of segmentation.
- **mIoU (mean Intersection over Union)**: Measures the overlap between predicted and actual segmentation areas.

| Model              | Accuracy | mIoU  |
|--------------------|----------|-------|
| UNet               | 64,377 | 64,533  |
| UNet++             | 67,677     | 62,876  |
| ViTSegmenter       | 60,683 | 64,791  |
| ResViTSegmenter    | 66,689     | 64,641  |

- **UNet++** achieved the **highest accuracy**, making it highly effective for accurate segmentation tasks.
- **ResViTSegmenter** demonstrated both **high accuracy and mIoU**, making it robust for tasks requiring a balance between accuracy and segmentation overlap.
- **UNet** and **ViTSegmenter** models had high mIoU values, though their accuracy was comparatively lower.


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gstaros/AerialSegmentation.gi
   cd satellite-image-segmentation
   
2. Run prepare_test_session.py. If you want to test This will copy the example dataset and create CSV files used in the training
   ```bash
   python prepare_test_session.py

3. Update config files:

   - _run/config.yml_ - models parameters
   - _run//train_config.yml_ - train parameters like batch_size, train, and validation CSV files location, checkpoints for resuming training
   - _run/interference_config.yml_ - interference parameters like batch_size, test CSV  files location


### Interference

To use those models in interference you need to download checkpoint files for all of the models you want to use into _src/checkpoints_ location. 
Checkpoints can be downloaded from this [link](https://drive.google.com/file/d/1eCu5kNiCRMJfx86gqvWgINDixZ49k5gm/view?usp=drive_link). 

### Examples

![image](https://github.com/user-attachments/assets/e75d4dc4-5009-4d6c-98d0-3f84eb725305)

![image](https://github.com/user-attachments/assets/24c88604-4b93-480f-b61f-62a856e5794e)

![image](https://github.com/user-attachments/assets/4d8778c6-b7ed-47b0-9d76-6fb44a08df9a)
