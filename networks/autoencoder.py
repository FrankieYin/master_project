import torch.nn as nn

from networks import SavableModule, device, LATENT_CODE_SIZE
from networks.point_net_encoder import PointNetfeat
from networks.sdf_net_decoder import SDFNet


class AutoEncoder(nn.Module):
    def __init__(self, encoder: PointNetfeat, decoder: SDFNet):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.to(device)

    def forward(self, batch):
        """
        :param point_sdf: [N, point_dim]
        :return:
        """
        num_pts = batch.shape[2]
        latent_code, input_trans = self.encoder(batch)

        # the decoder use a different tensor layout than the encoder, but I don't want to change it just in case I
        # mess up.
        latent_code_expanded = latent_code.view(-1, 1, LATENT_CODE_SIZE).repeat(1, num_pts, 1)
        sdf_pred = self.decoder(latent_code_expanded, batch.transpose(2, 1)) # [B, N, point_dim]

        return sdf_pred, input_trans, latent_code

    @property
    def device(self):
        return next(self.parameters()).device