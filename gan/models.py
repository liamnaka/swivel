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
#         self.fc3 = nn.Linear(1024, 1024)
#         self.bn3 = nn.BatchNorm1d(1024)
        self.fc_bias = nn.Linear(1024, self.num_vertices * 3)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
       
        self.bias_scale = bias_scale
        self.obj_scale = 1.0
        self.elu_alpha = 1.0
        self.v_relu = nn.LeakyReLU(0.001, inplace=True)

    def forward(self, z):
        """ Makes a forward pass with the given input through G.

        Arguments:
            z (tensor): input noise (e.g. images)
        """

        x = self.bn1(self.relu(self.fc1(z)))
        x = self.bn2(self.relu(self.fc2(x)))
#         x = self.bn3(self.relu(self.fc3(x)))
        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.num_vertices, 3)

        base = self.sphere_vs * self.obj_scale
        #base = self.xp.broadcast_to(base[None, :, :], bias.shape)

        sign = torch.sign(base)
        base = torch.abs(base)
        
        #base = torch.log(base/(1-base))
        
        #vertices = F.elu(base + bias, alpha=self.elu_alpha, inplace=True) + self.elu_alpha 
        
        #vertices = (-self.v_relu((-F.elu(base + bias, alpha=self.elu_alpha, inplace=True)) + 1) + 2) / 3
        
        #vertices = F.elu(base + bias, alpha=self.elu_alpha, inplace=True) + self.elu_alpha
        
        vertices = -self.v_relu(-(self.v_relu(base + bias) - 1.5 * self.obj_scale)) + 1.5 * self.obj_scale
        #vertices = torch.sigmoid(bias)
        
        #vertices = torch.relu((torch.tanh(bias) * self.obj_scale + base))
        #vertices -= vertices.mean(dim=1, keepdim=True)
       
        
        #print(vertices.shape)
        vertices = vertices * sign  #/ torch.max((torch.max(vertices) + self.eps), self.one)
        
       
    
        # return self.sphere_vs[None, :, :].expand(z.shape[0], -1, -1)
        return vertices
    
    
class ReverseGenerator(nn.Module):
    r""" Reconstruct z from mesh"""

    def __init__(self, batch_size=1, z_dim=512):
        super(ReverseGenerator, self).__init__()

        icosphere = tm.creation.icosphere(3, 1) # 642 vertice sphere,  rad=1
        self.sphere_vs = torch.from_numpy(icosphere.vertices).float().cuda()
        self.num_vertices = len(icosphere.vertices)
        self.fc1 = nn.Linear(1024,z_dim)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(self.num_vertices * 3, 1024)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
       
    def forward(self, mesh):
        """ Makes a forward pass with the given input through G.

        Arguments:
            z (tensor): input noise (e.g. images)
        """
        
        v_sign = torch.sign(mesh)
        v_abs = torch.abs(mesh)
        base_abs = torch.abs(self.sphere_vs)
        bias_abs = v_abs - base_abs
        bias = v_sign * bias_abs

        x = self.bn4(self.relu(self.fc4(bias.view(-1, self.num_vertices * 3))))
        x = self.bn3(self.relu(self.fc3(x)))
        x = self.bn2(self.relu(self.fc2(x)))
        reconstructed_z = self.fc1(x)
 
        return reconstructed_z


class Encoder(nn.Module):
    r""" Reconstruct z from Image"""

    def __init__(self, batch_size=1, z_dim=512, image_size=128, image_channels=4):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        
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
        self.latent_layer = nn.Linear(128 * ds_size ** 2, z_dim + 2)
        
    def forward(self, img):
        """ Makes a forward pass with the given input through RG.

        Arguments:
            z (tensor): input noise (e.g. images)
        """

        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        zs = self.latent_layer(out)
        latent = zs[:, :self.z_dim ]
        elevation = (torch.tanh(zs[:, self.z_dim ]) * 90).view(-1, 1) * 0.999
        azimuth = (torch.sigmoid(zs[:, self.z_dim +1]) * 360).view(-1, 1)

        return latent, elevation, azimuth


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

    def forward(self, z, elevations=None, azimuths=None):
        if self.random_pose:
            elev = torch.FloatTensor(self.batch_size, 1).uniform_(-90, 90)
            azim = torch.FloatTensor(self.batch_size, 1).uniform_(0, 360)
        else:
            elev = self.default_elevation
            azim = self.default_azimuth
            
#         if elevations is not None:
#             elev = elevations
#         if azimuths is not None:
#             azim = azimuths

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
