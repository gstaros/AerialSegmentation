import torch
import os

class SaveModel:
    def __init__(self, path, measure_type='epoch'):
        self.path = path
        self.measure_type = measure_type
        self.best_score = None

        assert measure_type in ['epoch', 'min', 'max']

    def __call__(self, epoch, model, optimizer, measure=None):

        # per epoch
        if self.measure_type == 'epoch':
            self.save_model(epoch, model, optimizer)
            print(f'Checkpoint saved for epoch {epoch}')


        # measure is minimized -> save model
        if self.measure_type == 'min':
            if self.best_score is None:
                self.best_score = measure
                self.save_model(epoch, model, optimizer)

            if measure < self.best_score:
                self.best_score = measure
                self.save_model(epoch, model, optimizer)
                print(f'Checkpoint saved for epoch {epoch} for value: {measure}')


        # measure is maximized -> save model
        if self.measure_type == 'max':
            if self.best_score is None:
                self.best_score = measure
                self.save_model(epoch, model, optimizer)

            if measure > self.best_score:
                self.best_score = measure
                self.save_model(epoch, model, optimizer, path=self.path)
                print(f'Checkpoint saved for epoch {epoch} for value: {measure}')


    def save_model(self, epoch, model, optimizer):

        if not os.path.exists(self.path):
            os.makedirs(f'{self.path}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'{self.path}/epoch_{epoch}.pth')