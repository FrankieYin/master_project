import os

import torch
import torch.nn as nn

LATENT_CODE_SIZE = 128
POINT_DIM = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SavableModule(nn.Module):
    def __init__(self, filename):
        super(SavableModule, self).__init__()
        self.filename = filename

    def get_filename(self, epoch=None, filename=None):
        if filename is None:
            filename = self.filename
        if epoch is None:
            return os.path.join(MODEL_PATH, filename)
        else:
            filename = filename.split('.')
            filename[-2] += '-epoch-{:05d}'.format(epoch)
            filename = '.'.join(filename)
            return os.path.join(CHECKPOINT_PATH, filename)

    def load(self, epoch=None):
        self.load_state_dict(torch.load(self.get_filename(epoch=epoch)), strict=False)

    def save(self, epoch=None):
        if epoch is not None and not os.path.exists(CHECKPOINT_PATH):
            os.mkdir(CHECKPOINT_PATH)
        torch.save(self.state_dict(), self.get_filename(epoch=epoch))

    @property
    def device(self):
        return next(self.parameters()).device