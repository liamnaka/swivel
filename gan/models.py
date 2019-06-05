import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh as tm

from soft_renderer.renderer import SoftRenderer

class MeshGenerator(nn.Module):
    r""" Generate Mesh from z by deforming sphere"""

    def __init__(self, batch_size=1, z_dim=512, bias_scale=1.0, centroid_scale=0.1):
        super(MeshGenerator, self).__init__()

        icosphere = tm.creation.icosphere(3, 1) # 642 vertice sphere,  rad=1
        self.num_vertices = len(icosphere.vertices)
        self.sphere_vs = torch.from_numpy(icosphere.vertices).float().cuda()
        self.sphere_fs = torch.from_numpy(icosphere.faces).int().cuda()
        # Repeat along batch dim
        self.sphere_fs = self.sphere_fs[None, :, :].expand(batch_size, -1, -1)

        self.fc1 = nn.Linear(z_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc_bias = nn.Linear(1024, self.num_vertices * 3)
        #self.fc_centroids = nn.Linear(1024, 3)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
       
        #self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 1.0
        self.elu_alpha = 1.0
        self.v_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, z):
        """ Makes a forward pass with the given input through G.

        Arguments:
            z (tensor): input noise (e.g. images)
        """

        x = self.bn1(self.relu(self.fc1(z)))
        x = self.bn2(self.relu(self.fc2(x)))
        #x = torch.tanh(self.fc_out(x))
        #x = x.view(-1, self.num_vertices, 3)
        #centroids = self.fc_centroids(x) * self.centroid_scale
        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.num_vertices, 3)

        base = self.sphere_vs * self.obj_scale
        #base = self.xp.broadcast_to(base[None, :, :], bias.shape)

        sign = torch.sign(base)
        base = torch.abs(base)
        #base = torch.log(base / (1 - base))
        
        #centroids = torch.tanh(centroids)
        #centroids = centroids[:, None, :].expand(*bias.shape)
        #scale_pos = 1 - centroids
        #scale_neg = centroids + 1
        #vertices = F.elu(base + bias, alpha=self.elu_alpha, inplace=True) + self.elu_alpha 
        vertices = -self.v_relu(-F.elu(base + bias, alpha=self.elu_alpha, inplace=True) + self.elu_alpha) 
        #print(vertices.shape)
        vertices = vertices * sign * 0.5 #/ torch.max((torch.max(vertices) + self.eps), self.one)
        #vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        #vertices += centroids
        #vertices *= 0.5
        
        
        #constrain to keep points within their quadrant
        #mask = torch.ge((self.sphere_vs + x) * self.sphere_vs, 0).float()
        #new_mesh_vs = (x + self.sphere_vs) * mask
    
        # return self.sphere_vs[None, :, :].expand(z.shape[0], -1, -1)
        return vertices#x.view(-1, self.num_vertices, 3)#new_mesh_vs


class DiffRenderer(nn.Module):
    r""" Wrapper class for Soft Rasterizer implemenation from SoftRas."""

    def __init__(self, image_size=128, background_color=[0,0,0],
                 texture_type='surface',
                 camera_mode='look_at_from', orig_size=128,
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
            super(DiffRenderer, self).__init__()
            self.renderer = SoftRenderer(
                image_size=image_size,
                background_color=background_color,
                texture_type=texture_type,
                camera_mode='look_at_from',
                near=0,
                orig_size=orig_size,
                light_mode=light_mode,
                light_intensity_ambient=light_intensity_ambient,
                light_color_ambient=light_color_ambient,
                light_intensity_directionals=light_intensity_directionals,
                light_color_directionals=light_color_directionals,
                light_directions=light_directions,
            )

    def forward(self, vertices, faces, elevations, azimuths, distance):
        return self.renderer(
            vertices,
            faces,
            elevations=elevations,
            azimuths=azimuths,
            distance=distance
        )


class RenderedGenerator(nn.Module):
    r""" Link together mesh generator and Soft Rasterizer to generate images"""
    def __init__(self, batch_size=32, z_dim=512, image_size=128,
                orig_size=64, background_color=[0,0,0], random_pose=True,
                default_elevation=30, default_azimuth=30, distance=1.5):
        super(RenderedGenerator, self).__init__()

        self.mesh_generator = MeshGenerator(batch_size=batch_size, z_dim=z_dim)
        self.sphere_fs = self.mesh_generator.sphere_fs
        self.renderer = DiffRenderer(
            image_size=image_size,
            orig_size=orig_size,
            background_color=background_color
        )
        self.random_pose = random_pose
        self.distance = distance
        self.batch_size = batch_size
        if not self.random_pose:
            self.default_elevation = torch.Tensor(
                [default_elevation]
            ).float().cuda()[None, :].expand(batch_size, -1)
            self.default_azimuth = torch.Tensor(
                [default_azimuth]
            ).float().cuda()[None, :].expand(batch_size, -1)

    def forward(self, z):
        if self.random_pose:
            elev = torch.FloatTensor(self.batch_size, 1).uniform_(-90, 90)
            azim = torch.FloatTensor(self.batch_size, 1).uniform_(0, 360)
        else:
            elev = self.default_elevation
            azim = self.default_azimuth

        vertices = self.mesh_generator(z)
        rendered_image  = self.renderer(
            vertices, self.sphere_fs, elev, azim, self.distance
        )

        return  rendered_image, vertices


class Discriminator(nn.Module):
    r""" From https://github.com/eriklindernoren/PyTorch-GAN/blob/master/
        implementations/infogan/infogan.py
    """

    def __init__(self, image_size=128, image_channels=4):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(image_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        scores = self.adv_layer(out)

        return scores
