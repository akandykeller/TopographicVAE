import torch
import kornia
from kornia.geometry import warp_perspective
import numpy as np
from torchvision import transforms
from math import pi

class To_Color(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        """
        """
        tensor_rgb = transforms.functional.to_tensor(tensor).repeat(3, 1, 1)
        tensor_rgb[1:] = 0.0

        return transforms.functional.to_pil_image(tensor_rgb)

    def __repr__(self):
        format_string = self.__class__.__name__ 
        return format_string

class AddRandomTransformationDims(object):
    def __init__(self, angle_set, color_set, scale_set):
        self.angle_set = angle_set
        self.color_set = color_set
        self.scale_set = scale_set

    def __call__(self, tensor):
        x = tensor.unsqueeze(0)
        b, c, h, w = x.shape

        # define the rotation center
        center = torch.ones(x.shape[0], 2, device=x.device)
        center[..., 0] = w / 2  # x
        center[..., 1] = h / 2  # y

        x_expanded = x.new_zeros((x.shape[0], len(self.angle_set), 1, 1, c, h, w))

        start_angle = torch.randint(len(self.angle_set), (1,))
        start_scale = torch.randint(len(self.scale_set), (1,))
        start_color = torch.randint(len(self.color_set), (1,))

        self.angle_set = self.angle_set[start_angle:] + self.angle_set[:start_angle]
        self.scale_set = self.scale_set[start_scale:] + self.scale_set[:start_scale]
        self.color_set = self.color_set[start_color:] + self.color_set[:start_color]

        transform_type = torch.randint(0, 3, (1,))

        if transform_type == 0:
            scale_idx = torch.randint(0, len(self.scale_set), (1,))
            color_idx = torch.randint(0, len(self.color_set), (1,))
            scale_set = [self.scale_set[scale_idx]]
            color_set = [self.color_set[color_idx]]
            angle_set = self.angle_set
        elif transform_type == 1:
            angle_idx = torch.randint(0, len(self.angle_set), (1,))
            color_idx = torch.randint(0, len(self.color_set), (1,))
            angle_set = [self.angle_set[angle_idx]]
            color_set = [self.color_set[color_idx]]
            scale_set = self.scale_set
        elif transform_type == 2:
            angle_idx = torch.randint(0, len(self.angle_set), (1,))
            scale_idx = torch.randint(0, len(self.scale_set), (1,))
            angle_set = [self.angle_set[angle_idx]]
            scale_set = [self.scale_set[scale_idx]]
            color_set = self.color_set

        for a_i, angle in enumerate(angle_set):
            for c_i, color in enumerate(color_set):
                for s_i, scale in enumerate(scale_set):
                    bsz_angles = torch.ones(x.shape[0]) * angle # * speed
                    bsz_colors = torch.ones(x.shape[0]) * color
                    bsz_scales = torch.ones(x.shape[0]) * scale

                    # compute the transformation matrix
                    M = kornia.get_rotation_matrix2d(center, bsz_angles, bsz_scales).to(x.device)

                    # apply the transformation to original image
                    x_t = kornia.warp_affine(x, M, dsize=(h, w))

                    if c == 3:
                        # Apply color rotation
                        x_t = kornia.color.adjust_hue(x_t, bsz_colors)
                    t_i = max(a_i, c_i, s_i)
                    x_expanded[:, t_i, :, :] = x_t

        tensor = x_expanded.squeeze(0)
        return tensor


class AddDualTransformationDims(object):
    def __init__(self, angle_set, color_set, scale_set):
        self.angle_set = angle_set
        self.color_set = color_set
        self.scale_set = scale_set

    def __call__(self, tensor):
        x = tensor.unsqueeze(0)
        b, c, h, w = x.shape

        # define the rotation center
        center = torch.ones(x.shape[0], 2, device=x.device)
        center[..., 0] = w / 2  # x
        center[..., 1] = h / 2  # y

        x_expanded = x.new_zeros((x.shape[0], len(self.angle_set), 1, 1, c, h, w))

        start_angle = torch.randint(len(self.angle_set), (1,))
        start_color = torch.randint(len(self.color_set), (1,))
        scale = 1.0

        self.angle_set = self.angle_set[start_angle:] + self.angle_set[:start_angle]
        self.color_set = self.color_set[start_color:] + self.color_set[:start_color]

        for t_i, (angle, color) in enumerate(zip(self.angle_set, self.color_set)):
            bsz_angles = torch.ones(x.shape[0]) * angle # * speed
            bsz_colors = torch.ones(x.shape[0]) * color
            bsz_scales = torch.ones(x.shape[0]) * scale

            # compute the transformation matrix
            M = kornia.get_rotation_matrix2d(center, bsz_angles, bsz_scales).to(x.device)

            # apply the transformation to original image
            x_t = kornia.warp_affine(x, M, dsize=(h, w))

            if c == 3:
                # Apply color rotation
                x_t = kornia.color.adjust_hue(x_t, bsz_colors)
            x_expanded[:, t_i, :, :] = x_t

        tensor = x_expanded.squeeze(0)
        return tensor


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (c, height, width) """

    def __init__(self, shape):
        self.bsz = shape[0]
        self.num_channels = shape[1]
        self.height = shape[2]
        self.width = shape[3]

    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, image_batch, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)
        M = torch.tensor(np.stack([mat] * self.bsz)).to(torch.float32)

        return warp_perspective(image_batch, M, (self.height, self.width))

    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))


class AddPerspectiveTransformationDims(object):
    def __init__(self, angle_set, color_set, scale_set, seq_len=18, shape=(1, 3, 28, 28)):
        self.seq_len = seq_len
        self.it = ImageTransformer(shape)
        self.angle_set = angle_set
        self.color_set = color_set
        self.scale_set = scale_set


    def __call__(self, tensor):
        x = tensor.unsqueeze(0)
        b, c, h, w = x.shape

        start_angle = torch.randint(len(self.angle_set), (1,))

        x_expanded = x.new_zeros((x.shape[0], len(self.angle_set), 1, 1, c, h, w))

        start_angle = torch.randint(len(self.angle_set), (1,))
        start_color = torch.randint(len(self.color_set), (1,))
        scale = 1.0

        self.angle_set = self.angle_set[start_angle:] + self.angle_set[:start_angle]
        self.color_set = self.color_set[start_color:] + self.color_set[:start_color]

        for t_i, (angle, color) in enumerate(zip(self.angle_set, self.color_set)):
            bsz_colors = torch.ones(x.shape[0]) * color

            # phi = 60*np.cos(np.radians(angle))
            # theta = 60*np.sin(np.radians(angle))

            if angle != 0.0:
                if angle == 180.0:
                    x_t = self.it.rotate_along_axis(x, phi=angle+1, gamma=angle+1)
                else:
                    x_t = self.it.rotate_along_axis(x, phi=angle, gamma=angle)
                # x_t = self.it.rotate_along_axis(x, theta=theta, phi=phi)
                # x_t = self.it.rotate_along_axis(x, phi=angle, dx = 5)
            else:
                x_t = x

            if c == 3:
                # Apply color rotation
                x_t = kornia.color.adjust_hue(x_t, bsz_colors)

            x_expanded[:, t_i, :, :] = x_t

        tensor = x_expanded.squeeze(0)
        return tensor
