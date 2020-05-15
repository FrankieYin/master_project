import json
import math
import time
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import deep_sdf
from train_deep_sdf import get_learning_rate_schedules
from networks.deep_sdf_decoder import Decoder
from networks.sdf_net_decoder import SDFNet
import torch.utils.data as data_utils
from train_autoencoder import SDFSampleDataset


def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)


def save_checkpoint(epoch, decoder, optimiser, lat_vecs, training_loss, experiment, filename='latest'):
    if not os.path.exists(f'checkpoints/{experiment}'): os.mkdir(f'checkpoints/{experiment}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'latent_state_dict': lat_vecs.state_dict(),
        'loss': training_loss,
    }, f'checkpoints/{experiment}/{filename}.pth')

def plot_loss(experiment, epoch=None):
    latest = 'latest'
    experiment_path = f'checkpoints/{experiment}'
    checkpoint = torch.load(f'{experiment_path}/{epoch if epoch else latest}.pth')
    loss = checkpoint['loss']
    plt.plot(loss)
    plt.show()

def train_decoder():
    experiment = 'train_decoder'
    specs = json.load(open('examples/chairs/specs.json'))
    lr_schedules = get_learning_rate_schedules(specs)
    latent_size = 128
    hidden_dim = latent_size * 2
    num_samp_per_scene = 16384
    scene_per_batch = 5
    code_bound = 1
    clamp_dist = 0.1
    code_reg_lambda = 1e-4

    decoder = SDFNet(latent_size).to('cuda')

    split_file = json.load(open('5_sample.json'))
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

    num_scenes = len(sdf_dataset)

    lat_vecs = nn.Embedding(num_scenes, latent_size, max_norm=code_bound).to('cuda')
    nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        1 / math.sqrt(latent_size),
    )
    loss_l1 = torch.nn.L1Loss(reduction="sum")  # TODO: why sum?
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    start_epoch = 1
    num_epochs = 500
    batch_split = 1

    training_loss = []
    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        start_time = time.time()
        running_loss = []
        adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        for sdf_data, indices in sdf_loader:
            sdf_data = sdf_data.to('cuda')
            sdf_data = sdf_data.reshape(-1, 4)
            num_sdf_samples = sdf_data.shape[0]
            sdf_data.requires_grad = False
            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
            indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1).to('cuda')
            optimizer_all.zero_grad()

            batch_vecs = lat_vecs(indices)
            input = torch.cat([batch_vecs, xyz], dim=1)

            # NN optimization
            pred_sdf = decoder(input)
            # pred_sdf = decoder(batch_vecs, xyz)
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
        # save_checkpoint(epoch, decoder, optimizer_all, lat_vecs, training_loss, experiment)
        if epoch % 100 == 0:
            save_checkpoint(epoch, decoder, optimizer_all, lat_vecs, training_loss, experiment, filename=str(epoch))

if __name__ == '__main__':
    train_decoder()
    plot_loss('train_decoder', 500)




