import json
import math
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import deep_sdf
from networks import device, POINT_DIM
from networks.autoencoder import AutoEncoder
from networks.point_net_encoder import SimplePointnet, PointNetfeat
from networks.sdf_net_decoder import SDFNet
import torch.utils.data as data_utils

from train_deep_sdf import get_learning_rate_schedules, StepLearningRateSchedule

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
        return sample, idx

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

def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

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


def save_checkpoint(epoch, encoder, decoder, optimiser, training_loss, experiment, filename='latest'):
    if not os.path.exists(f'checkpoints/{experiment}'): os.mkdir(f'checkpoints/{experiment}')
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
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

def plot_loss(experiment, epoch=None):
    latest = 'latest'
    experiment_path = f'checkpoints/{experiment}'
    checkpoint = torch.load(f'{experiment_path}/{epoch if epoch else latest}.pth')
    loss = checkpoint['loss']
    plt.plot(loss)
    plt.show()

def train_autoencoder(experiment):
    specs = json.load(open('examples/chairs/specs.json'))
    lr_schedules = get_learning_rate_schedules(specs)
    latent_size = 128
    point_dim = 4
    hidden_dim = latent_size * 2
    num_samp_per_scene = 16384
    scene_per_batch = 8
    num_epochs = 2000
    code_bound = 1
    clamp_dist = 0.1
    code_reg_lambda = 1e-4

    encoder = SimplePointnet(c_dim=latent_size, dim=point_dim).to('cuda')
    decoder = SDFNet(latent_size).to('cuda')

    split_file = json.load(open('examples/splits/sv2_chairs_train.json'))
    sdf_dataset = deep_sdf.data.SDFSamples(
        'data', split_file, subsample=num_samp_per_scene, load_ram=False  # num_samp_per_scene (1 scene = 1 shape)
    )
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,  # scene_per_batch
        shuffle=True,
        num_workers=0,  # num_data_loader_threads
        drop_last=True,
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    if os.path.exists(f'checkpoints/{experiment}/latest.pth'):
        checkpoint = torch.load(f'checkpoints/{experiment}/latest.pth')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_all.load_state_dict(checkpoint['optimiser_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        training_loss = checkpoint['loss']
        print(f"Resuming training at epoch {start_epoch}")
    else:
        start_epoch = 1
        training_loss = []
        print('Starting training')

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()
        running_loss = []
        adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        for sdf_data, indices in tqdm(sdf_loader):
            optimizer_all.zero_grad()
            sdf_data = sdf_data.to('cuda')
            batch_vecs = encoder(sdf_data)
            batch_vecs = batch_vecs.view(-1, 1, latent_size).repeat(1, num_samp_per_scene, 1).view(-1, latent_size)

            sdf_data = sdf_data.reshape(-1, 4)
            num_sdf_samples = sdf_data.shape[0]
            sdf_data.requires_grad = False
            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
            input = torch.cat([batch_vecs, xyz], dim=1)

            # NN optimization
            pred_sdf = decoder(input)
            pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

            batch_loss = loss_l1(pred_sdf, sdf_gt) / num_sdf_samples
            l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
            reg_loss = (code_reg_lambda * min(1, epoch / 100) * l2_size_loss) / num_sdf_samples
            batch_loss += reg_loss
            batch_loss.backward()
            optimizer_all.step()

            running_loss.append(batch_loss.item())

        epoch_duration = time.time() - start_time
        epoch_loss = np.mean(running_loss)
        training_loss.append(epoch_loss)

        print("Epoch {:d}, {:.1f}s. Loss: {:.8f}".format(epoch, epoch_duration, epoch_loss))

        # always save the latest snapshot
        save_checkpoint(epoch, encoder, decoder, optimizer_all, training_loss, experiment)
        if epoch % 100 == 0:
            save_checkpoint(epoch, encoder, decoder, optimizer_all, training_loss, experiment, filename=str(epoch))

if __name__ == '__main__':
    # train_autoencoder('train_autoencoder')
    plot_loss('train_autoencoder')
