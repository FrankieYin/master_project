#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os

import pyrender
import torch
import trimesh
from tqdm import tqdm

import deep_sdf
import deep_sdf.metrics
import deep_sdf.workspace as ws


def evaluate():

    split_file = json.load(open('examples/splits/sv2_chairs_test2.json'))
    shape_names = split_file['ShapeNetV2']['03001627']
    chamfer_results = []
    for shape_name in tqdm(shape_names):
        reconstructed_mesh_filename = f'checkpoints/train_autoencoder/reconstructions2/{shape_name}.ply'
        ground_truth_samples_filename =  f'data/SurfaceSamples/ShapeNetV2/03001627/{shape_name}.ply'
        normalization_params_filename = f'data/NormalizationParameters/ShapeNetV2/03001627/{shape_name}.npz'

        ground_truth_points = trimesh.load(ground_truth_samples_filename)
        try:
            reconstruction = trimesh.load(reconstructed_mesh_filename)
        except ValueError:
            # the reconstruction mesh does not exist
            continue
        normalization_params = np.load(normalization_params_filename)

        chamfer_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
            ground_truth_points,
            reconstruction,
            normalization_params["offset"],
            normalization_params["scale"],
        )

        chamfer_results.append([shape_name, chamfer_dist])

    torch.save(chamfer_results, 'checkpoints/train_autoencoder/chamfer_k.pth')

def evaluate_unknown():

    split_file = json.load(open('examples/splits/sv2_chairs_test3.json'))
    shape_names = split_file['ShapeNetV2']['03001627']
    chamfer_results = []
    for shape_name in tqdm(shape_names):
        reconstructed_mesh_filename = f'checkpoints/train_autoencoder/reconstructions/{shape_name}.ply'
        ground_truth_samples_filename =  f'data/SurfaceSamples/ShapeNetV2/03001627/{shape_name}.ply'
        normalization_params_filename = f'data/NormalizationParameters/ShapeNetV2/03001627/{shape_name}.npz'

        ground_truth_points = trimesh.load(ground_truth_samples_filename)
        try:
            reconstruction = trimesh.load(reconstructed_mesh_filename)
        except ValueError:
            # the reconstruction mesh does not exist
            continue
        normalization_params = np.load(normalization_params_filename)

        chamfer_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
            ground_truth_points,
            reconstruction,
            normalization_params["offset"],
            normalization_params["scale"],
        )

        chamfer_results.append([shape_name, chamfer_dist])

    torch.save(chamfer_results, 'checkpoints/train_autoencoder/chamfer_u.pth')


def view_mesh(mesh_filename, rotate_axis=[0, 0, 1]):
    # mesh_filename = f'checkpoints/train_autoencoder/{recon}/{meshanme}.ply'
    mesh = trimesh.load(mesh_filename)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True, rotate_axis=rotate_axis)

if __name__ == "__main__":

    # evaluate()
    # evaluate_unknown()
    # for file in ['chamfer_k', 'chamfer_u']:
    #     chamfer = torch.load(f'checkpoints/train_autoencoder/{file}.pth')
    #     chamfer.sort(key=lambda x: x[1])
    view_mesh('checkpoints/train_autoencoder/reconstructions2/2972fd770304663cb3d180f4523082e1.ply')