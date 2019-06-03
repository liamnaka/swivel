import matplotlib.pyplot as plt
import os
import time
import tqdm
import numpy as np
import imageio
import argparse

import torch

import soft_renderer as sr

from models import RenderedGenerator, Discriminator, MeshGenerator, DiffRenderer


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
        default=os.path.join(data_dir, 'obj/spot/spot_triangulated.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()

    batch_size = 64

    # other settings
    camera_distance = 1.5
    elevation = torch.zeros((batch_size,1)) + 15
    #elevation[0] = 30
    azimuth = torch.zeros((batch_size,1))

    z_dim = 512
    z= torch.FloatTensor(batch_size, z_dim).uniform_(-1, 1).cuda()
    # load from Wavefront .obj file
    generator = MeshGenerator(batch_size, z_dim)
    renderer = DiffRenderer(image_size=128)

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        renderer.cuda()

    vertices = generator(z)
    faces = generator.sphere_fs

    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation-icosphere-gen.gif'), mode='I')
    for num, azimuth_val in enumerate(loop):
        loop.set_description('Drawing rotation of rendered generator')
        # renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        azimuth += 4
        if num == 0:
            t0 = time.time()
        images = renderer(vertices, faces, elevations=elevation, azimuths=azimuth, distance=camera_distance)
        if num == 0:
            t1 = time.time()
        image = images.detach().cpu().numpy()[31].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()
    print(t1-t0)

    # save to textured obj
    mesh = sr.Mesh(vertices, faces)
    mesh.save_obj(os.path.join(args.output_dir, 'saved_icosphere.obj'), save_texture=False)


if __name__ == '__main__':
    main()
