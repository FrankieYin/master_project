import json

import torch.nn as nn
import torch
import torch.utils.data as data
import numpy as np
import os
from networks.point_net_encoder import PointNetfeat
from networks.sdf_net_decoder import SDFNet
from networks.autoencoder import AutoEncoder
from networks import LATENT_CODE_SIZE
import pyrender

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

        return samples

    def _remove_nans(self, tensor):
        tensor_nan = torch.isnan(tensor[:, 3])
        return tensor[~tensor_nan, :]

if __name__ == '__main__':
    encoder = PointNetfeat(use_global_feat=True, feature_transform=False)
    # decoder = SDFNet()
    # model = AutoEncoder(encoder, decoder)
    #
    # dataset = SDFSampleDataset('data/SdfSamples/ShapeNetV2/03001627/', 'test.json')
    # print(len(dataset))
    # for i in range(3):
    #     sample = dataset[i]
    #     print(sample.shape)
    #
    # batch_size = 8
    # train_data = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #
    # for i_batch, batch in enumerate(train_data):
    #     num_pts = batch.shape[2]
    #     latent_code, input_trans = encoder(batch)
    #     latent_code = latent_code.view(-1, LATENT_CODE_SIZE, 1).repeat(1, 1, num_pts)
    #     break

    batch = torch.randn(8, 100, 4)
    latent_code, input_trans = encoder(batch)

    # latent_code = torch.randn(8, 128, 100)
    # batch = torch.randn(8, 4, 100)
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
