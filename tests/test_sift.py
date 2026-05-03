import torch
import numpy as np

from models.siftNet import (
    HistogramLayer,
    SIFTNet,
    SIFTOrientationLayer,
    SubGridAccumulationLayer,
    get_sift_subgrid_coords,
    get_siftnet_features,
)


def test_angles_to_vectors_2d_pytorch():
    """
    Make sure that pi/4, pi, -pi, 0 radians are matched to correct (dx,dy) values.
    """
    import numpy as np
    import torch

    from models.siftNet import angles_to_vectors_2d_pytorch

    angles = torch.from_numpy(np.array([np.pi / 4, np.pi, -np.pi, 0]))
    vectors_2d = angles_to_vectors_2d_pytorch(angles)
    gt_vectors_2d = torch.tensor(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-1.0, 0], [-1.0, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(gt_vectors_2d, vectors_2d, atol=1e-3)


def test_HistogramLayer():
    """
    Convert a Tensor of shape (1, 10, 4, 1) to a Tensor with shape (1, 8, 4, 1), which
    represents a per-pixel histogram.
    """
    hist_layer = HistogramLayer()
    x = torch.tensor(
        [
            [
                [[0.1962], [19.6157], [0.2494], [24.9441]],
                [[0.1111], [16.6294], [0.0585], [29.4236]],
                [[-0.0390], [3.9018], [-0.1667], [16.6671]],
                [[-0.1663], [-11.1114], [-0.2942], [-5.8527]],
                [[-0.1962], [-19.6157], [-0.2494], [-24.9441]],
                [[-0.1111], [-16.6294], [-0.0585], [-29.4236]],
                [[0.0390], [-3.9018], [0.1667], [-16.6671]],
                [[0.1663], [11.1114], [0.2942], [5.8527]],
                [[0.1962], [16.6294], [0.2942], [16.6671]],
                [[0.0390], [11.1114], [-0.0585], [24.9441]],
            ]
        ]
    )
    per_px_hist = hist_layer(x)
    f = per_px_hist.sum(dim=2)
    gt_f = torch.tensor(
        [
            [
                [20.2000],
                [30.0000],
                [0.0000],
                [0.0000],
                [0.0000],
                [0.0000],
                [0.0000],
                [0.3000],
            ]
        ]
    )
    assert torch.allclose(f, gt_f, atol=1e-3)


def test_SIFTOrientationLayer():
    """
    Take a Tensor of shape (1, 2, 4, 1) representing the image gradients Ix, Iy
    of a (4 x 1) image, and produce the orientation and magnitude information
    at each pixel, e.g. producing a tensor of shape (1, 10, 4, 1).
    """
    so_layer = SIFTOrientationLayer()
    im_grads = torch.tensor(
        [
            [
                [[0.1962], [16.6294], [0.2942], [16.6671]],
                [[0.0390], [11.1114], [-0.0585], [24.9441]],
            ]
        ]
    )
    dot_products = so_layer(im_grads)

    # Ground truth (10,4) tensor, with 10 values for each of the 4 pixels.
    gt_dot_products = torch.tensor(
        [
            [0.1962, 19.6157, 0.2494, 24.9441],
            [0.1111, 16.6294, 0.0585, 29.4236],
            [-0.0390, 3.9018, -0.1667, 16.6671],
            [-0.1663, -11.1114, -0.2942, -5.8527],
            [-0.1962, -19.6157, -0.2494, -24.9441],
            [-0.1111, -16.6294, -0.0585, -29.4236],
            [0.0390, -3.9018, 0.1667, -16.6671],
            [0.1663, 11.1114, 0.2942, 5.8527],
            [0.1962, 16.6294, 0.2942, 16.6671],
            [0.0390, 11.1114, -0.0585, 24.9441],
        ]
    )
    assert torch.allclose(dot_products.squeeze(), gt_dot_products, atol=1e-3)


def test_SIFTNet():
    """ """
    img_bw = np.arange(256).reshape(16, 16)  # numpy array
    img_bw = torch.from_numpy(img_bw)
    img_bw = img_bw.unsqueeze(0).unsqueeze(0)
    img_bw = img_bw.float()

    net = SIFTNet()
    per_px_8dim_feat = net(img_bw)
    assert [1, 8, 17, 17] == [per_px_8dim_feat.shape[i] for i in range(4)]
    assert torch.allclose(
        per_px_8dim_feat.detach().sum(), torch.tensor(768899.5000), atol=5
    )


def test_SubGridAccumulationLayer():
    """
    Convert [1, 8, 5, 6] to [1, 8, 6, 7]
    """
    sg_acc_layer = SubGridAccumulationLayer()
    per_px_histogram = torch.zeros((1, 8, 5, 6))

    for i in range(5):
        for j in range(6):
            per_px_histogram[0, :, i, j] = torch.arange(i, i + 8)

    accum_hists = sg_acc_layer(per_px_histogram)

    assert accum_hists.shape == (1, 8, 6, 7)
    gt_accum_hists_sum = torch.tensor(
        [
            [
                [72.0, 108.0, 144.0, 144.0, 144.0, 108.0, 72.0],
                [108.0, 162.0, 216.0, 216.0, 216.0, 162.0, 108.0],
                [144.0, 216.0, 288.0, 288.0, 288.0, 216.0, 144.0],
                [180.0, 270.0, 360.0, 360.0, 360.0, 270.0, 180.0],
                [216.0, 324.0, 432.0, 432.0, 432.0, 324.0, 216.0],
                [252.0, 378.0, 504.0, 504.0, 504.0, 378.0, 252.0],
                [288.0, 432.0, 576.0, 576.0, 576.0, 432.0, 288.0],
                [324.0, 486.0, 648.0, 648.0, 648.0, 486.0, 324.0],
            ]
        ]
    )
    assert torch.allclose(accum_hists.sum(dim=2), gt_accum_hists_sum, atol=5)


def test_get_sift_subgrid_coords():
    """
    Ensure that given the center point of a 16x16 patch, we pull out the accumulated
            values for each of the 16 subgrids. We verify that the 16 x- and y-coordinates
            are matched with one and only 4x4 subgrid.
    """
    x_center = 12
    y_center = 11
    # We use integers because these come from valid coordinates in the image,
    # not sub-pixel approximations.
    x_grid_coords, y_grid_coords = get_sift_subgrid_coords(x_center, y_center)

    assert x_grid_coords.dtype in [np.int32, np.int64]
    assert y_grid_coords.dtype in [np.int32, np.int64]

    assert x_grid_coords.shape == (16,)
    assert y_grid_coords.shape == (16,)

    def count_coords_between(x_coords, y_coords, x_start, x_end, y_start, y_end):
        """
        Args:
        -	x_coords: Numpy array of shape (N,)
        -	y_coords: Numpy array of shape (N,)
        -	x_start: int
        -	x_end: int
        -	y_start: int
        -	y_end: int

        Returns:
        -	count of how many 2d coordinates lie within given range
        """
        x_logicals = np.logical_and(x_start <= x_coords, x_coords < x_end)
        y_logicals = np.logical_and(y_start <= y_coords, y_coords < y_end)
        return np.logical_and(x_logicals, y_logicals).sum()

    x_start = x_center - 8
    y_start = y_center - 8
    # ensure one (x,y) coord exists in each central 2x2 subgrid of
    # each 4x4 subgrid (center is ambiguous)
    for i in range(4):
        for j in range(4):
            x_2x2_s = x_start + (i * 4) + 1
            x_2x2_e = x_2x2_s + 2
            y_2x2_s = y_start + (j * 4) + 1
            y_2x2_e = y_2x2_s + 2

            count = count_coords_between(
                x_grid_coords, y_grid_coords, x_2x2_s, x_2x2_e, y_2x2_s, y_2x2_e
            )

            # print(f'{x_2x2_s} <= x < {x_2x2_e}, {y_2x2_s} <= y < {y_2x2_e}')
            assert count == 1


def test_get_siftnet_features():
    """ """
    x = np.array([8, 8, 7, 9])  # numpy array
    y = np.array([7, 9, 8, 8])  # numpy array
    img_bw = np.arange(256).reshape(16, 16)  # numpy array
    img_bw = torch.from_numpy(img_bw)
    img_bw = img_bw.unsqueeze(0).unsqueeze(0)
    img_bw = img_bw.float()

    features = get_siftnet_features(img_bw, x, y)

    assert np.allclose(features.sum(), 22.039, atol=1)
    assert features.shape == (4, 128)

    # gt_feat_crop = np.array(
    # 	[
    # 		[0.28135952, 0.20184263],
    # 		[0.28796558, 0.17183169],
    # 		[0.27522191, 0.12444288],
    # 		[0.        , 0.23030874]
    # 	])
    # assert np.allclose(features[:,64:66], gt_feat_crop)
