import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

"""
Image gradients are needed for both SIFT and the Harris Corner Detector, so we
implement the necessary code only once, here.
"""


def get_sobel_xy_parameters() -> torch.nn.Parameter:
    """
    Populate the conv layer weights for the Sobel layer (image gradient
    approximation).

    There should be two sets of filters: each should have size (1 x 3 x 3)
    for 1 channel, 3 pixels in height, 3 pixels in width. When combined along
    the batch dimension, this conv layer should have size (2 x 1 x 3 x 3), with
    the Sobel_x filter first, and the Sobel_y filter second.

    Returns:
        kernel: Torch parameter representing (2, 1, 3, 3) conv filters
    """
    Sobel_x = (
        torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
    )
    Sobel_y = (
        torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
    )
    kernel = nn.Parameter(torch.cat((Sobel_x, Sobel_y)))
    return kernel


def get_gaussian_kernel(ksize: int = 7, sigma: float = 5) -> torch.nn.Parameter:
    """
    Generate a Gaussian kernel to be used in HarrisNet for calculating a
    second moment matrix (SecondMomentMatrixLayer).

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: torch.nn.Parameter of size [ksize, ksize]
    """
    gauss_1d = np.zeros((ksize, 1))
    total, index = 0, 0
    for i in range(-int(np.floor(ksize / 2)), int(np.floor(ksize / 2)) + 1):
        x1 = 1 / np.sqrt(2 * np.pi * sigma**2)
        x2 = np.exp(-(i**2) / (2 * sigma**2))
        g = x1 * x2
        gauss_1d[index] = g
        index += 1
        total += g
    kernel = np.outer(gauss_1d, gauss_1d) / total**2
    kernel = nn.Parameter(torch.from_numpy(kernel).float())
    return kernel

class ChannelProductLayer(torch.nn.Module):
    """
    ChannelProductLayer: Compute I_xx, I_yy and I_xy,

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing I_xx, I_yy and I_xy respectively.
    """

    def __init__(self):
        super(ChannelProductLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of the previous layer, which is of size
        (num_image x 2 x width x height) for Ix and Iy.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for I_xx, I_yy and I_xy.

        HINT: you may find torch.cat(), torch.mul() useful here
        """

        num_image, _, height, width = x.shape
        output = torch.zeros([num_image, 3, height, width])
        for i in range(num_image):
            image_gx = x[i][0]
            image_gy = x[i][1]
            I_xx = torch.mul(image_gx, image_gx).unsqueeze(0)
            I_yy = torch.mul(image_gy, image_gy).unsqueeze(0)
            I_xy = torch.mul(image_gx, image_gy).unsqueeze(0)
            output[i] = torch.cat((I_xx, I_yy, I_xy))
        return output

class SecondMomentMatrixLayer(torch.nn.Module):
    """
    SecondMomentMatrixLayer: Given a 3-channel image I_xx, I_xy, I_yy, then
    compute S_xx, S_yy and S_xy.

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing S_xx, S_yy and S_xy, respectively

    """

    def __init__(self, ksize: int = 7, sigma: float = 5):
        """
        You may find get_gaussian_kernel() useful. You must use a Gaussian
        kernel with filter size `ksize` and standard deviation `sigma`. After
        you pass the unit tests, feel free to experiment with other values.

        Args:
        -   None

        Returns:
        -   None
        """
        super(SecondMomentMatrixLayer, self).__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.kernel = get_gaussian_kernel(self.ksize, self.sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of previous layer, which is of size
        (num_image, 3, width, height) for I_xx and I_yy and I_xy.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for S_xx, S_yy and S_xy

        HINT:
        - You can either use your own implementation from project 1 to get the
        Gaussian kernel, OR reimplement it in get_gaussian_kernel().
        """

        num_image, _, height, width = x.shape
        output = torch.zeros([num_image, 3, height, width])
        kernel = self.kernel
        k = kernel.shape[0]
        kernel = torch.reshape(kernel, (1, 1, k, k))

        for i in range(num_image):
            I_xx = x[i][0].unsqueeze(0).unsqueeze(0)
            I_yy = x[i][1].unsqueeze(0).unsqueeze(0)
            I_xy = x[i][2].unsqueeze(0).unsqueeze(0)

            S_xx = F.conv2d(input=I_xx, weight=kernel.float(), padding=k // 2)
            S_yy = F.conv2d(input=I_yy, weight=kernel.float(), padding=k // 2)
            S_xy = F.conv2d(input=I_xy, weight=kernel.float(), padding=k // 2)

            output[i][0] = S_xx
            output[i][1] = S_yy
            output[i][2] = S_xy
        return output

class CornerResponseLayer(torch.nn.Module):
    """
    Compute R matrix.

    The output is a tensor of size (num_image, channel, height, width),
    represent corner score R

    HINT:
    - For matrix A = [a b;
                      c d],
      det(A) = ad-bc, trace(A) = a+d
    """

    def __init__(self, alpha: float = 0.05):
        """
        Don't modify this __init__ function!
        """
        super(CornerResponseLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass to compute corner score R

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        You may find torch.mul() useful here.
        """
        num_image, _, height, width = x.shape
        output = torch.zeros([num_image, 1, height, width])
        for i in range(num_image):
            S_xx = x[i][0]
            S_yy = x[i][1]
            S_xy = x[i][2]
            det = torch.mul(S_xx, S_yy) - torch.mul(S_xy, S_xy)
            trace = S_xx + S_yy
            R = det - self.alpha * (trace**2)
            output[i] = R
        return output

class NMSLayer(torch.nn.Module):
    """
    NMSLayer: Perform non-maximum suppression,

    the output is a tensor of size (num_image, 1, height, width),

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d
    """

    def __init__(self):
        super(NMSLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum, and return this binary
        image, multiplied with the cornerness response values. We'll be testing
        only 1 image at a time. Input and output will be single channel images.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        (Potentially) useful functions: nn.MaxPool2d, torch.where(), torch.median()
        """
        num_image, c, height, width = x.shape
        zero_tensor = torch.zeros([c, height, width])
        output = torch.zeros([num_image, 1, height, width])

        for i in range(num_image):
            image = x[i]
            thresh = torch.median(image)
            image = torch.where(image >= thresh, image, zero_tensor)
            pool = torch.nn.MaxPool2d(kernel_size=7, padding=7 // 2, stride=1)
            maximums = pool(image)
            output[i] = torch.where(image == maximums, image, zero_tensor)
        return output


class ImageGradientsLayer(torch.nn.Module):
    """
    ImageGradientsLayer: Compute image gradients Ix & Iy. This can be
    approximated by convolving with Sobel filter.
    """

    def __init__(self):
        super(ImageGradientsLayer, self).__init__()

        # Create convolutional layer
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            bias=False,
            padding=(1, 1),
            padding_mode="zeros",
        )

        # Instead of learning weight parameters, here we set the filter to be
        # Sobel filter
        self.conv2d.weight = get_sobel_xy_parameters()
        self.conv2d.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of ImageGradientsLayer. We'll test with a
        single-channel image, and 1 image at a time (batch size = 1).

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, (num_image, 2, height, width)
            tensor for Ix and Iy, respectively.
        """
        return self.conv2d(x)
