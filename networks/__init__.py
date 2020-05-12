import torch

LATENT_CODE_SIZE = 128
POINT_DIM = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
