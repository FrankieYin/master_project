import torch.nn as nn

from networks.point_net_encoder import PointNetfeat
from networks.sdf_net_decoder import SDFNet


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, point_sdf):
        """
        :param point_sdf: [N, point_dim]
        :return:
        """
        return self.decoder(self.encoder(point_sdf))

