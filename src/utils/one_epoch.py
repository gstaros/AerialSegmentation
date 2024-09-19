from collections.abc import Iterable
from tqdm import tqdm
import torch


def train_one_epoch(epoch, model, train_dataloader, criterion, optimizer, device, writer):
    running_train_loss = 0.

    num_examples = len(train_dataloader)

    for X_train, y_train in tqdm(train_dataloader):
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()

        pred_mask = model(X_train)

        if isinstance(criterion, Iterable):
            combined_train_loss = 0.
            for loss_fn in criterion:
                combined_train_loss += loss_fn(pred_mask, y_train)
            train_loss = combined_train_loss
        else:
            train_loss = criterion(pred_mask, y_train)

        train_loss.backward()
        optimizer.step()

        running_train_loss += train_loss

    # get average loss per this epoch
    train_loss_out = running_train_loss / num_examples

    writer.add_scalar('Loss/train', train_loss_out, epoch)

    return train_loss_out


def val_one_epoch(epoch, model, val_dataloader, criterion, optimizer, device, writer):
    running_val_loss = 0.

    num_examples = len(val_dataloader)

    with torch.no_grad():
        for X_val, y_val in tqdm(val_dataloader):
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            pred_mask = model(X_val)

            if isinstance(criterion, Iterable):
                combined_val_loss = 0.
                for loss_fn in criterion:
                    combined_val_loss += loss_fn(pred_mask, y_val)
                val_loss = combined_val_loss
            else:
                val_loss = criterion(pred_mask, y_val)

            running_val_loss += val_loss


    # get average loss per this epoch
    val_loss_out = running_val_loss / num_examples

    writer.add_scalar('Loss/val', val_loss_out, epoch)

    return val_loss_out