import math

import numpy as np
import scipy.misc
import os
from PIL import Image

IMAGE_SIZE = 64
DISTANCE = 2.0

import soft_renderer as sr

def run():
    #class_ids = [
    #    '02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459', '04090263',
    #    '04256520', '04379243', '04401088', '04530566']
    class_ids = ['02691156']
    # directory_shapenet_id = '../../resource/shapenetcore_ids'
    directory_rendering = '/shapenet/shapenet_images_%d_%.1f/%s/%s'
    shapenet_root = '/shapenet/shapenet-v2-airplanes'
    filename_shapenet_obj = os.path.join(shapenet_root, '%s/%s/model.obj')

    renderer = sr.SoftRenderer(
        image_size=IMAGE_SIZE, camera_mode="look_at_from", perspective=False
    )
    renderer.cuda()

    # ce33bf3ec6438e5bef662d1962a11f02
    for class_id in class_ids:
        # ids = open(os.path.join(directory_shapenet_id, '%s_trainids.txt' % class_id)).readlines()
        # ids += open(os.path.join(directory_shapenet_id, '%s_valids.txt' % class_id)).readlines()
        # ids += open(os.path.join(directory_shapenet_id, '%s_testids.txt' % class_id)).readlines()
        obj_ids = [x[0] for x in os.walk(os.path.join(shapenet_root, class_id)]
        num_obj = len(obj_ids)
        for i, obj_id in enumerate(obj_ids):
            if i % 100 == 0:
                print("Progress: ", i, "/", num_obj, " objects rendered.")
            print('rendering: %s %d / %d' % (class_id, i, len(obj_ids)))

            directory = directory_rendering % (IMAGE_SIZE, DISTANCE, class_id, obj_id)
            directory_tmp = directory + '_'
            if os.path.exists(directory):
                continue
            if os.path.exists(directory_tmp):
                continue
            try:
                os.makedirs(directory_tmp)
            except:
                continue

            mesh = sr.Mesh.from_obj(
                filename_shapenet_obj % (class_id, obj_id),
                load_texture=False, normalization=True, texture_type='surface'
            )
            seed = int(int(obj_id, 16) ** (1/5))
            np.random.seed(seed)
            azimuth = torch.Tensor([[np.random.random() * 360]]).cuda()
            elevation = torch.Tensor(
                [[(np.random.random() * 180) - 90.0]]
            ).cuda()
            elevation *= 0.99999 #Avoid -90 and 90
            image = renderer.render_mesh(
                mesh, elevations=elevation, azimuths=azimuth, distance=DISTANCE
            ).detach().cpu().numpy()[0].transpose((1, 2, 0))
            filename = os.path.join(
                directory, 'e%03d_a%03d.png' % (elevation, azimuth)
            )
            save = Image.fromarray((image * 255).astype('uint8'), 'RGB')
            im.save(filename)

            try:
                os.rename(directory_tmp, directory)
            except:
                continue


run()
