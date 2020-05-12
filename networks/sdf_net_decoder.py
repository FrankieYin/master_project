import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from networks import SavableModule, LATENT_CODE_SIZE, POINT_DIM

SDF_NET_BREADTH = 256

class SDFNet(SavableModule):
    def __init__(self, latent_code_size=LATENT_CODE_SIZE, device='cuda'):
        super(SDFNet, self).__init__(filename="sdf_net_decoder")
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

        self.to(device)

    def forward(self, latent_codes, points):
        """
        :param latent_codes: [N, LATENT_CODE_DIM]
        :param points: [N, POINT_DIM]
        :return:
        """
        input = torch.cat([latent_codes, points], dim=1)
        x = self.layers1(input)
        x = torch.cat((x, input), dim=1)
        x = self.layers2(x)
        return x.squeeze()