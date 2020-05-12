import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from networks import SavableModule, LATENT_CODE_SIZE, POINT_DIM

SDF_NET_BREADTH = 256

class SDFNet(nn.Module):
    def __init__(self, latent_code_size=LATENT_CODE_SIZE):
        super(SDFNet, self).__init__()
        self.layers1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(POINT_DIM + latent_code_size, SDF_NET_BREADTH)),
            nn.ReLU(inplace=True),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(inplace=True),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(inplace=True),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(inplace=True)
        )

        self.layers2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH + latent_code_size + POINT_DIM,SDF_NET_BREADTH)),
            nn.ReLU(inplace=True),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(inplace=True),

            nn.utils.weight_norm(nn.Linear(SDF_NET_BREADTH, SDF_NET_BREADTH)),
            nn.ReLU(inplace=True),

            nn.Linear(SDF_NET_BREADTH, 1),
            nn.Tanh()
        )

    def forward(self, latent_codes, points):
        """
        :param latent_codes: [B, N, LATENT_CODE_DIM]
        :param points: [B, N, POINT_DIM]
        :return:
        """
        input = torch.cat([latent_codes, points], dim=2)
        x = self.layers1(input)
        x = torch.cat((x, input), dim=2)
        x = self.layers2(x)
        return x.squeeze()