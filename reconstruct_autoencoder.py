import json
import logging
import os
import time

import plyfile
import pyrender
import skimage.measure
import torch
import numpy as np

import deep_sdf
from train_autoencoder import load_or_init_model, SDFSampleDataset, CLAMP_DIST

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

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()
    running_loss = []

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        running_loss.append(loss.item())

        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        # if e % 50 == 0:
        #     logging.debug(loss.cpu().data.numpy())
        #     logging.debug(e)
        #     logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    print("loss from deepsdf: {}".format(np.mean(running_loss)))
    return loss_num, latent



def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    latent_repeat = latent_vector.expand(num_samples, -1)

    sdf = decoder(latent_repeat.unsqueeze(0), queries.unsqueeze(0))

    return sdf.unsqueeze(1) # this is so the shape matches that of the original code: [num_points, 1]


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

def loss_fn(sdf_gt, sdf_pred):
    sdf_gt.clamp(-CLAMP_DIST, CLAMP_DIST)
    sdf_pred.clamp(-CLAMP_DIST, CLAMP_DIST)

    l1_loss = torch.mean(torch.abs(sdf_pred - sdf_gt))
    return l1_loss

def deepsdf_reconstruct(instance_name):
    specs = json.load(open('examples/chairs/specs.json'))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = 256

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            'examples/chairs/ModelParameters', 'latest' + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    full_filename = f'data/SdfSamples/ShapeNetV2/03001627/{instance_name}.npz'
    data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
    data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
    data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

    err, latent = reconstruct(
        decoder,
        int(800),
        latent_size,
        data_sdf,
        0.01,  # [emp_mean,emp_var],
        0.1,
        num_samples=8000,
        lr=5e-3,
        l2reg=True,
    )

    print(f'latent norm: {latent.norm()}')

    # decoder.eval()
    # with torch.no_grad():
    #     deep_sdf.create_mesh(
    #         decoder, latent, 'deepsdf_mesh', N=256, max_batch=int(2 ** 18)
    #     )

    return latent


if __name__ == '__main__':
    model, optimiser, epoch, training_loss = load_or_init_model('changed_decoder_input_dim')
    dataset = SDFSampleDataset('data/SdfSamples/ShapeNetV2/03001627/', 'test.json')
    sample = dataset[32].unsqueeze(0)  # [1, point_dim, N]
    instance_name = dataset.split_file[32]
    print(instance_name)

    # scene = pyrender.Scene()
    # scene.add(pyrender.Mesh.from_points(sample.squeeze().transpose(0, 1)[:, :3]))
    # pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

    mesh_filename = 'mesh'

    latent_code = deepsdf_reconstruct(instance_name)

    with torch.no_grad():
        model.eval()
        sample = sample.to('cuda')
        sdf_pred, input_trans, latent_code = model(sample)
        print(f'my latent norm: {latent_code.norm()}')
        sdf_gt = sample[:, -1, :].squeeze()
        l1_loss = loss_fn(sdf_gt, sdf_pred)
        print(l1_loss)
    #
    #     # latent_code, input_trans = model.encode(sample.to('cuda'))
    #     create_mesh(model.decoder, latent_code, mesh_filename, max_batch=int(2 ** 18))