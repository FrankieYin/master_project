import torch.nn as nn

from networks import device, LATENT_CODE_SIZE
from networks.point_net_encoder import PointNetfeat
from networks.sdf_net_decoder import SDFNet


class AutoEncoder(nn.Module):
    def __init__(self, encoder: PointNetfeat, decoder: SDFNet):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.to(device)

    # helper functions: give access to encoder and decoder
    def encode(self, batch):
        return self.encoder(batch)

    def decode(self, latent_code, batch):
        return self.decoder(latent_code, batch)

    def forward(self, batch):
        """
        :param batch: [B, point_dim, N]
        :return:
        """
        num_pts = batch.shape[2]
        latent_code, input_trans = self.encode(batch)

        # the decoder use a different tensor layout than the encoder, but I don't want to change it just in case I
        # mess up.
        # print(latent_code.shape)
        latent_code_expanded = latent_code.view(-1, 1, LATENT_CODE_SIZE).repeat(1, num_pts, 1)
        sdf_pred = self.decode(latent_code_expanded, batch[:, :3, :].transpose(2, 1)) # [B, N, 3] since we only need xyz

        return sdf_pred, input_trans, latent_code

    @property
    def device(self):
        return next(self.parameters()).device