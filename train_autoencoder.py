import json

import torch.nn as nn
import torch
import torch.utils.data as data
import numpy as np
import os
from networks.point_net_encoder import PointNetfeat
from networks.sdf_net_decoder import SDFNet
from networks.autoencoder import AutoEncoder
from networks import LATENT_CODE_SIZE, device, POINT_DIM
import pyrender

CLAMP_DIST = 0.1
SIGMA = 0.01


class SDFSampleDataset(data.Dataset):
    def __init__(self, sdf_sample_dir, split_file_path, subsample=16384):
        with open(split_file_path, 'r') as f:
            self.split_file = json.load(f)['ShapeNetV2']['03001627']
        self.sdf_sample_dir = sdf_sample_dir
        self.subsample = subsample

    def __len__(self):
        return len(self.split_file)

    def __getitem__(self, idx):
        filename = os.path.join(self.sdf_sample_dir, self.split_file[idx] + '.npz')
        sample = self._unpack_sdf_samples(filename, self.subsample)
        return sample

    def _unpack_sdf_samples(self, filename, subsample):
        npz = np.load(filename)

        pos_tensor = self._remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = self._remove_nans(torch.from_numpy(npz["neg"]))

        # split the sample into half
        half = int(subsample / 2)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples = torch.cat([sample_pos, sample_neg], 0)

        return samples.view(samples.shape[1], samples.shape[0])

    def _remove_nans(self, tensor):
        tensor_nan = torch.isnan(tensor[:, 3])
        return tensor[~tensor_nan, :]


if __name__ == '__main__':
    encoder = PointNetfeat(use_global_feat=True, feature_transform=False)
    decoder = SDFNet()
    model = AutoEncoder(encoder, decoder)

    #
    dataset = SDFSampleDataset('data/SdfSamples/ShapeNetV2/03001627/', 'test.json')
    # print(len(dataset))
    # for i in range(3):
    #     sample = dataset[i]
    #     print(sample.shape)
    #
    batch_size = 8
    train_data = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #
    for i_batch, batch in enumerate(train_data):
        batch = batch.to(device)  # [B, point_dim, N]
        sdf_pred, input_trans, latent_code = model(batch)
        sdf_gt = batch[:, -1, :].squeeze()

        sdf_gt.clamp(-CLAMP_DIST, CLAMP_DIST)
        sdf_pred.clamp(-CLAMP_DIST, CLAMP_DIST)

        l1_loss = torch.mean(torch.abs(sdf_pred - sdf_gt))
        latent_code_regulariser = (1 / SIGMA) * torch.mean(torch.norm(latent_code, dim=1))  # DeepSDF, eqn. 9
        # PointNet eqn. 2
        I = torch.eye(POINT_DIM)[None, :, :].to(device)
        feature_transform_regulariser = \
            torch.mean(torch.norm(torch.bmm(input_trans, input_trans.transpose(2, 1)) - I, dim=(1, 2)))

        loss = l1_loss + latent_code_regulariser + feature_transform_regulariser
        break

    # latent_code = torch.randn(8, 100, 128).to(device)
    # batch = torch.randn(8, 100, 4).to(device)
    # out = decoder(latent_code, batch)

    # try loading the data first
    # right now i dont care about testing set or train/test split just yet
    # focus on this:
    # load 100 training point into a point_cloud: [100, 4, num_points]
    # instance_name = '1aa07508b731af79814e2be0234da26c'
    # npz = np.load('data/SdfSamples/ShapeNetV2/03001627/{}.npz'.format(instance_name))
    # pos = npz['pos'][:, :3]
    # neg = npz['neg'][:, :3]
    #
    # points = np.vstack([pos, neg])
    #
    # scene = pyrender.Scene()
    # scene.add(pyrender.Mesh.from_points(points))
    # pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
