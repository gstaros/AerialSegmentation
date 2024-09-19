from torchmetrics.classification import MulticlassJaccardIndex
from .utils import one_hot_decoding
import torch



def evaluate_model(model, data_loader, device, num_classes):

    model.to(device)
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().

    correct = 0
    total = 0

    iou_metric = MulticlassJaccardIndex(num_classes).to(device)

    iou = 0.

    with torch.no_grad():
        for X_test, y_test in data_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            preds = one_hot_decoding(model(X_test).squeeze(), dim=1)
            correct += (preds == y_test).sum()
            total += preds.numel()

            iou += iou_metric(preds, y_test)

    accuracy = correct / total
    weighted_iou = iou / len(data_loader)

    return accuracy, weighted_iou