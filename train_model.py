
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import yaml
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import pandas as pd

from src.data import ImageSegmentationDataset

from src.utils.early_stopping import EarlyStopping
from src.utils.evaluate import evaluate_model
from src.utils.loss import DiceLoss, FocalLoss
from src.models.load_model import load_model
from src.utils.num_workers import find_num_workers
from src.utils.one_epoch import train_one_epoch, val_one_epoch
from src.utils.save_model import SaveModel


##### LOAD CONFIGS #####
with open('src/config/train_config.yml', 'r') as file:
    train_parameters = yaml.safe_load(file)


with open('src/config/config.yml', 'r') as file:
    model_config = yaml.safe_load(file)


##### TRAIN PARAMETERS #####
MODEL_SELECTION = train_parameters['MODEL_SELECTION']
LOAD_CHECKPOINT = train_parameters['LOAD_CHECKPOINT']
MODEL_SAVE_PATH = train_parameters['MODEL_SAVE_PATH']
SAVE_MODEL_TYPE = train_parameters['SAVE_MODEL_TYPE']

INPUT_CHANNELS  = model_config['INPUT_CHANNELS']
OUTPUT_CHANNELS = model_config['OUTPUT_CHANNELS']
IMAGE_SIZE = model_config['IMAGE_SIZE']

BATCH_SIZE = train_parameters['BATCH_SIZE']
LEARNING_RATE = train_parameters['LEARNING_RATE']
EPOCHS = train_parameters['EPOCHS']

TRAIN_CSV = train_parameters['TRAIN_CSV']
VAL_CSV = train_parameters['VAL_CSV']
START_EPOCH = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##### MODEL PARAMETERS #####
IMAGE_SIZE = model_config['IMAGE_SIZE']
INPUT_CHANNELS = model_config['INPUT_CHANNELS']
OUTPUT_CHANNELS = model_config['OUTPUT_CHANNELS']




##### DATASET & DATALOADER #####
transform = T.Compose([
    T.Resize([IMAGE_SIZE, IMAGE_SIZE], interpolation=T.InterpolationMode.NEAREST)
])


train_dataset = ImageSegmentationDataset(
    dir_file= train_parameters['TRAIN_CSV'],
    n_channels = INPUT_CHANNELS,
    n_classes = OUTPUT_CHANNELS,
    transform = transform # add more transform functions
)

val_dataset = ImageSegmentationDataset(
    dir_file= train_parameters['VAL_CSV'],
    n_channels = INPUT_CHANNELS,
    n_classes = OUTPUT_CHANNELS,
    transform = transform # add more transform functions
)


if train_parameters['FIND_NUM_WORKERS']:
    NUM_WORKERS = find_num_workers(val_dataset, BATCH_SIZE, 12)
else:
    NUM_WORKERS = 2


train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS )


##### CREATE MODEL #####
model = load_model(MODEL_SELECTION, model_config).to(device)

if LOAD_CHECKPOINT:
    checkpoint = torch.load(model_config['MODEL'][MODEL_SELECTION])
    START_EPOCH = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded')

criterion = [DiceLoss().to(device), FocalLoss(0.8, 2).to(device)]
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5)                                          



# Writer
writer = SummaryWriter(f'.runs/SatelliteSegmentation_{MODEL_SELECTION}')
start = datetime.now().strftime("%Y%m%d_%H%M%S")



# Early stopping
early_stopping = EarlyStopping(patience=15, delta=0.01)
save_model = SaveModel(f'{MODEL_SAVE_PATH}/{MODEL_SELECTION}/{start}', measure_type=SAVE_MODEL_TYPE)


best_acc = -1

train_stats = {'Loss': [], 'PixelAccuracy': [], 'IoU': []}
val_stats = {'Loss': [], 'PixelAccuracy': [], 'IoU': []}
lr_stats = []

for epoch in range(START_EPOCH, EPOCHS+1):
    print(f'Epoch {epoch}/{EPOCHS} - learning rate: {scheduler.get_last_lr()}')
    lr_stats.append(scheduler.get_last_lr()[0])
    ##### TRAINING #####

    # 1. Loss
    model.train()
    train_loss = train_one_epoch(epoch, model, train_dataloader, criterion, optimizer, device, writer)

    # 2. Accuracy
    train_acc, train_iou = evaluate_model(model, train_dataloader, device, OUTPUT_CHANNELS)
    writer.add_scalar('PixelAccuracy/train', train_acc, epoch)
    writer.add_scalar('IoU/train', train_iou, epoch)

    print(f'Training loss: {train_loss:.4f} | Traing acc: {train_acc*100:.2f}% | Training IoU: {train_iou:.6f}')
    train_stats['Loss'].append(train_loss.cpu().detach().numpy())
    train_stats['PixelAccuracy'].append(train_acc.cpu().detach().numpy())
    train_stats['IoU'].append(train_iou.cpu().detach().numpy())

    ##### VALIDATION #####

    # 1. Loss
    model.eval()
    val_loss = val_one_epoch(epoch, model, val_dataloader, criterion, optimizer, device, writer)

    # 2. Accuracy
    val_acc, val_iou = evaluate_model(model, val_dataloader, device, OUTPUT_CHANNELS)
    writer.add_scalar('PixelAccuracy/val', val_acc, epoch)
    writer.add_scalar('IoU/val', val_iou, epoch)
    print(f'Validation loss: {val_loss:.4f} | Validation acc: {val_acc*100:.2f}% | Validation IoU: {val_iou:.6f}')
    val_stats['Loss'].append(val_loss.cpu().detach().numpy())
    val_stats['PixelAccuracy'].append(val_acc.cpu().detach().numpy())
    val_stats['IoU'].append(val_iou.cpu().detach().numpy())

    ##### SAVING MODEL #####

    save_model(epoch, model, optimizer, val_loss)


    ##### SCHEDULER #####

    scheduler.step(val_loss)


    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping. Loading best model")
        early_stopping.load_best_model(model)
        print(f'Saving best model')
        torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/{MODEL_SELECTION}/{start}/LowestValLoss.pth')
        break

    print('-------------------')

writer.flush()

writer.close()


train_stats_df = pd.DataFrame(train_stats)
val_stats_df = pd.DataFrame(val_stats)

train_stats_df.to_csv(f'{MODEL_SAVE_PATH}/{MODEL_SELECTION}/{start}/train_stats.csv', index=True)
val_stats_df.to_csv(f'{MODEL_SAVE_PATH}/{MODEL_SELECTION}/{start}/val_stats.csv', index=True)

print("Loss and Accuracy was saved to CSV")