"""
Demo render.
1. save / load textured .obj file
2. render using SoftRas with different sigma / gamma
"""
import matplotlib.pyplot as plt
import os
import time
import tqdm
import numpy as np
import imageio
import argparse

import soft_renderer as sr

import torch


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
        default=os.path.join(data_dir, 'obj/spot/spot_triangulated.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()

    # other settings
    camera_distance = 2.0
    elevation = torch.zeros((1,1))
    elevation[0] = 30
    azimuth = torch.zeros((1,1))

    # load from Wavefront .obj file
    mesh = sr.Mesh.from_obj(args.filename_input,
                            load_texture=True, texture_res=5, texture_type='surface')

    # create renderer with SoftRas
    renderer = sr.SoftRenderer(camera_mode='look_at_from')

    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation-azim.gif'), mode='I')
    for num, azimuth_val in enumerate(loop):
        # rest mesh to initial state
        mesh.reset_()
        loop.set_description('Drawing azim rotation')
        # renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        azimuth[0] = azimuth_val
        if num == 0:
            t0 = time.time()
        images = renderer.render_mesh(mesh, elevations=elevation, azimuths=azimuth, distance=camera_distance)
        if num == 0:
            t1 = time.time()
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()
    print(t1-t0)
    
    # draw object from different view
    loop = tqdm.tqdm(list(range(-90, 90, 6)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation-elev.gif'), mode='I')
    for num, elev_val in enumerate(loop):
        # rest mesh to initial state
        mesh.reset_()
        loop.set_description('Drawing elev rotation')
        # renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        elevation[0] = elev_val
        if num == 0:
            t0 = time.time()
        images = renderer.render_mesh(mesh, elevations=elevation, azimuths=azimuth, distance=camera_distance)
        if num == 0:
            t1 = time.time()
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()
    print(t1-t0)

    # draw object from different sigma and gamma
    loop = tqdm.tqdm(list(np.arange(-4, -2, 0.2)))
    # renderer.transform.set_eyes_from_angles(camera_distance, elevation, 45)
    azimuth[0] = 45
    writer = imageio.get_writer(os.path.join(args.output_dir, 'bluring.gif'), mode='I')
    for num, gamma_pow in enumerate(loop):
        # rest mesh to initial state
        mesh.reset_()
        renderer.set_gamma(10**gamma_pow)
        renderer.set_sigma(10**(gamma_pow - 1))
        loop.set_description('Drawing blurring')
        images = renderer.render_mesh(mesh, elevations=elevation, azimuths=azimuth, distance=camera_distance)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

    # save to textured obj
    mesh.reset_()
    mesh.save_obj(os.path.join(args.output_dir, 'saved_spot.obj'), save_texture=True)


if __name__ == '__main__':
    main()
