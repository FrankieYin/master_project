import json
import random

import torch
import numpy as np

import deep_sdf
from networks.deep_sdf_decoder import Decoder
from networks.point_net_encoder import SimplePointnet
from networks.sdf_net_decoder import SDFNet


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    # init latent codes
    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else: # or else i can calculate the empirical stat
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss() # with no aggregation

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).to('cuda')
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)
        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)
        inputs = torch.cat([latent_inputs, xyz], 1).to('cuda')
        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        loss_num = loss.item()

    return loss_num, latent

if __name__ == '__main__':
    experiment = 'train_autoencoder'
    experiment_path = f'checkpoints/{experiment}'
    latent_size = 128
    point_dim = 4
    hidden_dim = latent_size * 2
    num_samp_per_scene = 16384
    scene_per_batch = 5
    code_bound = 1
    clamp_dist = 0.1
    code_reg_lambda = 1e-4

    encoder = SimplePointnet(c_dim=latent_size, dim=point_dim).to('cuda')
    decoder = SDFNet(latent_size).to('cuda')

    split_file = json.load(open('5_sample.json'))

    checkpoint = torch.load(f'{experiment_path}/500.pth')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    shape_names = split_file['ShapeNetV2']['03001627']
    shape_filenames = [f'data/SdfSamples/ShapeNetV2/03001627/{shape_name}.npz' for shape_name in shape_names]
    random.shuffle(shape_filenames)

    mesh_to_reconstruct = shape_filenames[0]
    npz = np.load(mesh_to_reconstruct)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    data_sdf = [pos_tensor, neg_tensor]
    data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
    data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]
    data_sdf = torch.cat(data_sdf, dim=0).unsqueeze(0).to('cuda') # [1, N, point_dim]

    with torch.no_grad():
        encoder.eval()
        latent = encoder(data_sdf)
        print("creating mesh")
        decoder.eval()
        deep_sdf.mesh.create_mesh(
            decoder, latent, f'{experiment_path}/mesh', N=256, max_batch=int(2 ** 18)
        )

    torch.save(latent, f'{experiment_path}/latent.pth')