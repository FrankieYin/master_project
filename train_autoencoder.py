import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks import device, POINT_DIM
from networks.autoencoder import AutoEncoder
from networks.point_net_encoder import PointNetfeat
from networks.sdf_net_decoder import SDFNet

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


def l1_loss(sdf_gt, sdf_pred):
    sdf_gt.clamp(-CLAMP_DIST, CLAMP_DIST)
    sdf_pred.clamp(-CLAMP_DIST, CLAMP_DIST)

    loss = torch.mean(torch.abs(sdf_pred - sdf_gt))
    return loss


def criterion(sdf_gt, sdf_pred, input_trans, latent_code):
    reconstruction_loss = l1_loss(sdf_gt, sdf_pred)

    # TODO: if sigma doesn't work try sigma^2
    latent_code_regulariser = SIGMA * torch.mean(torch.norm(latent_code, dim=1))  # DeepSDF, eqn. 9
    # PointNet eqn. 2
    I = torch.eye(POINT_DIM)[None, :, :].to(device)
    feature_transform_regulariser = \
        torch.mean(torch.norm(torch.bmm(input_trans, input_trans.transpose(2, 1)) - I, dim=(1, 2)))

    loss = reconstruction_loss + latent_code_regulariser + feature_transform_regulariser
    return loss


def save_checkpoint(epoch, model, optimiser, training_loss, experiment, filename='latest'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'loss': training_loss,
    }, f'checkpoints/{experiment}/{filename}.pth')


def load_or_init_model(experiment, epoch=None):
    experiment_path = f'checkpoints/{experiment}'
    encoder = PointNetfeat(use_global_feat=True, feature_transform=False)
    decoder = SDFNet()
    model = AutoEncoder(encoder, decoder)
    optimiser = optim.Adam(model.parameters(), lr=1e-5)

    if not epoch and not os.path.exists(experiment_path):
        # init model
        os.mkdir(experiment_path)
        return model, optimiser, 1, []

    elif not epoch:
        # load the latest.pth
        checkpoint = torch.load(f'{experiment_path}/latest.pth')
    else:
        checkpoint = torch.load(f'{experiment_path}/{epoch}.pth')

    epoch = checkpoint['epoch']
    training_loss = checkpoint['loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    return model, optimiser, epoch, training_loss


def train(experiment, num_epochs=100):
    model, optimiser, start_epoch, training_loss = load_or_init_model(experiment)

    dataset = SDFSampleDataset('data/SdfSamples/ShapeNetV2/03001627/', 'test.json')
    batch_size = 8
    train_data = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # training loop starts
    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()
        running_loss = []

        for i_batch, batch in tqdm(enumerate(train_data)):
            optimiser.zero_grad()
            batch = batch.to(device)  # [B, point_dim, N]
            sdf_pred, input_trans, latent_code = model(batch)
            sdf_gt = batch[:, -1, :].squeeze()

            loss = criterion(sdf_gt, sdf_pred, input_trans, latent_code)
            loss.backward()
            optimiser.step()
            running_loss.append(loss.item())

        epoch_duration = time.time() - start_time
        epoch_loss = np.mean(running_loss)
        training_loss.append(epoch_loss)

        print("Epoch {:d}, {:.1f}s. Loss: {:.8f}".format(epoch, epoch_duration, epoch_loss))

        # always save the latest snapshot
        save_checkpoint(epoch, model, optimiser, training_loss, experiment)
        if epoch % 10 == 0:
            save_checkpoint(epoch, model, optimiser, training_loss, experiment, filename=str(epoch))

def plot_loss(experiment):
    experiment_path = f'checkpoints/{experiment}'
    checkpoint = torch.load(f'{experiment_path}/latest.pth')
    loss = checkpoint['loss']
    plt.plot(loss)
    plt.show()

if __name__ == '__main__':
    experiment = '5_samples_latent_128_no_reg'
    num_epochs = 500

    encoder = PointNetfeat(use_global_feat=True, feature_transform=False)
    decoder = SDFNet()
    model = AutoEncoder(encoder, decoder)
    optimiser = optim.Adam(model.parameters(), lr=1e-5)
    # model, optimiser, start_epoch, training_loss = load_or_init_model(experiment)
    dataset = SDFSampleDataset('data/SdfSamples/ShapeNetV2/03001627/', '5_sample.json')
    batch_size = 5
    train_data = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # training loop starts
    training_loss = []
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        running_loss = []

        for i_batch, batch in tqdm(enumerate(train_data)):
            optimiser.zero_grad()
            batch = batch.to(device)  # [B, point_dim, N]
            sdf_pred, input_trans, latent_code = model(batch)
            sdf_gt = batch[:, -1, :].squeeze()

            loss = l1_loss(sdf_gt, sdf_pred) # TODO: experiment with only the l1 loss
            loss += SIGMA**2 * min(1, epoch / 100) * torch.mean(torch.norm(latent_code, dim=1))
            loss.backward()
            optimiser.step()
            running_loss.append(loss.item())

        epoch_duration = time.time() - start_time
        epoch_loss = np.mean(running_loss)
        training_loss.append(epoch_loss)

        print("Epoch {:d}, {:.1f}s. Loss: {:.8f}".format(epoch, epoch_duration, epoch_loss))

        if epoch_loss < 0.02:
            save_checkpoint(epoch, model, optimiser, training_loss, experiment, filename='sub002')

        # always save the latest snapshot
        save_checkpoint(epoch, model, optimiser, training_loss, experiment)
        if epoch % 100 == 0:
            save_checkpoint(epoch, model, optimiser, training_loss, experiment, filename=str(epoch))

