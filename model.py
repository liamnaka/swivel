import torch
import torch.nn as nn

from im2mesh.pix2mesh.models import Decoder
from im2mesh.encoder.pix2mesh_cond import Pix2mesh_Cond

# Camera distance is given

class CameraMatrixNet(nn.module):
        r''' Predict extrinsic camera matrix from feature maps of the first
        block of convolutional layers from the Pix2Mesh encoder. 128 x 28 x 28
        input size
        '''

    def __init__(self, camera_distance):
        super().__init__()

        actvn = nn.ReLU()
        num_fm = 16 # int(512/32)
        self.camera_distance = camera_distance
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_fm*4, num_fm*8, 3, stride=2, padding=1), actvn,
            nn.Conv2d(num_fm*2, num_fm*8, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm, num_fm*8, 3, stride=1, padding=1))
        self.linear = nn.Linear(num_fm x 28 x 28, 2)

    def forward(self, fm0):
        batch_size = fm0.shape[0]

        cam_params = self.linear2(self.linear1(self.conv_block(fm0)))

        cam_params_in_degrees = nn.functional.Tanh(cam_params) * 360
        azimuths = cam_params_in_degrees[:, 0]
        angles = cam_params_in_degrees[:, 1]

        cam_mat = can_params.view(-1, 3, 4)

        return cam_mat, azimuth, angle, distance


class MeshGenerator(nn.module):
    r""" Wrapper class for Pixel2Mesh implemenation from im2mesh. With additional camera paramter estimator"""

    def __init__(self, hidden_dim=192, feat_dim=1280):
        super().__init__()

        ellipsoid = pickle.load(
            open('im2mesh/pix2mesh/ellipsoid/info_ellipsoid.dat', 'rb'),
            encoding='latin1'
        )

        self.decoder = Decoder(
            self.ellipsoid, hidden_dim=hidden_dim, feat_dim=feat_dim)
        self.encoder = Pix2mesh_Cond(return_feature_maps=True)
        self.cam_mat_net = CameraMatrixNet()

    def forward(self, x):
        """ Makes a forward pass with the given input through the network.

        Arguments:
            x (tensor): input tensors (e.g. images)
            fm (tensor): feature maps from the conditioned network
        """
        fm = self.encoder(x)
        amera_mat = self.cam_mat_net(fm[0])
        outputs_1, outputs_2 = self.decoder(x, fm, camera_mat)
        return outputs_1, outputs_2, camera_mat

class Renderer(nn.module):
    r""" Wrapper class for Soft Rasterizer implemenation from SoftRas."""

    # camera_mode='look_at'


# Img -> Mesh (Pix2Mesh)

# Mesh -> Img (SoftRas)
