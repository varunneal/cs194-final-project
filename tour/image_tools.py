import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.color import rgba2rgb, rgb2gray
import scipy.interpolate as skinterp
from skimage.draw import polygon


def load_rgb(pth):
    # Loads image as rgb
    im = cv2.imread(pth, 1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def load_gray(im_path):
    # Returns 2-channel grayscale image
    im = cv2.imread(im_path, 0)
    return im


def load_rgba(im_path):
    # Returns 4-channel rgba image.
    # Accepts source in format rgb or rgba.
    # If source is rgb, alpha channel is added.
    im = plt.imread(im_path)
    return add_alpha(im)


def to_gray(im):
    if len(im.shape) == 2:
        return im
    elif im.shape[2] == 3:
        return rgb2gray(im)
    elif im.shape[2] == 4:
        return rgb2gray(rgba2rgb(im))


def NCC(v1, v2):
    """
    returns NCC of 1d arrays v1 and v2.
    """
    nv1 = v1 / np.linalg.norm(v1)
    nv2 = v2 / np.linalg.norm(v2)
    return np.dot(nv1, nv2)


def lattice(guess):
    """
    returns all points in a +/-5 window around guess
    """

    return np.array([
        [int(h + guess[0]), int(w + guess[1])]
        for h in range(-5, 6)
        for w in range(-5, 6)
    ])


def gkern(kernlen, std=None):
    """
    Returns a 2D Gaussian kernel array.
    """
    if not std:
        std = 0.3 * ((kernlen - 1) * 0.5 - 1) + 0.8
    gkern1d = cv2.getGaussianKernel(kernlen, std)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def blur(img, amount, std=None):
    """
    Applies a gaussian kernal to img with size amount and std.
    """
    G = gkern(amount, std)
    if len(img.shape) == 2:
        return convolve2d(img, G, mode='same')
    r = convolve2d(img[:, :, 0], G, mode='same')
    g = convolve2d(img[:, :, 1], G, mode='same')
    b = convolve2d(img[:, :, 2], G, mode='same')

    return normalize(np.dstack([r, g, b]), hard=True)


def normalize(img):
    """
    normalizes img to [0, 1]
    """
    if np.max(img) == np.min(img):
        return img
    if len(img.shape) == 2:
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[2]):
        layer = img[:, :, i]
        if np.max(layer) == np.min(layer):
            out[:, :, i] = layer
        else:
            out[:, :, i] = (layer - np.min(layer)) / (np.max(layer) - np.min(layer))
    return out


def add_alpha(img):
    """
    adds an alpha channel to rgb img
    """
    if img.shape[2] == 3:
        alpha = np.ones((img.shape[0], img.shape[1]))
        return np.dstack((img, alpha))
    return img


def points(im):
    """
    returns all the indices in a given image as
    points in the format points_r, points_c
    with the corners as the first four entries
    """
    corners = np.array([
        [im.shape[0] - 1, 0, 0, im.shape[0] - 1],
        [0, 0, im.shape[1] - 1, im.shape[1] - 1],
        [1, 1, 1, 1]
    ])
    inner_points = np.array([[c, h, 1] for h in np.arange(im.shape[1]) for c in np.arange(im.shape[0])]).T

    return np.hstack((corners, inner_points))


def computeH(correspondences):
    """
    solves for h where A * h = b is our homography system of equations
    """
    num_points = correspondences.shape[1]
    A = np.zeros((2 * num_points, 8))
    b = np.zeros((2 * num_points, 1))
    for i in range(num_points):
        p0 = correspondences[:3, i]  # [x, y, 1]
        p1 = correspondences[3:6, i]  # [x', y', 1]
        A[2 * i, 0:3] = p0
        A[2 * i, 6:8] = [-p0[0] * p1[0], - p0[1] * p1[0]]
        A[1 + 2 * i, 3:6] = p0
        A[1 + 2 * i, 6:8] = [-p0[0] * p1[1], - p0[1] * p1[1]]

        b[2 * i] = p1[0]
        b[1 + 2 * i] = p1[1]

    h = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.vstack((h, [1])).reshape(3, 3)

    return H


def interp2(warped_r, warped_c, output_quad_r, output_quad_c, im0, im0_points):
    """
    Naive, nearest-neighbor, interpolation function, going channel-by-channel.
    """
    rgb = im0[im0_points[0, :], im0_points[1, :]]

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    intp_r = skinterp.griddata((warped_r, warped_c), r, (output_quad_r, output_quad_c), method="nearest")
    intp_g = skinterp.griddata((warped_r, warped_c), g, (output_quad_r, output_quad_c), method="nearest")
    intp_b = skinterp.griddata((warped_r, warped_c), b, (output_quad_r, output_quad_c), method="nearest")
    intp_a = np.ones_like(intp_r)
    return np.dstack((intp_r, intp_g, intp_b, intp_a))



def warp(img, correspondence):
    """
    Returns the warp of the image img to match the shape of
    the given correspondence matrix correspondence.
    """
    img_points = points(img)

    corr_matrix = computeH(correspondence)
    img_warped_points = np.matmul(corr_matrix, img_points)

    warped_r = np.int0(img_warped_points[0, :] / img_warped_points[2, :])
    warped_c = np.int0(img_warped_points[1, :] / img_warped_points[2, :])

    # Shift the transformed points so that they are all positive
    shift_r = - min(np.min(warped_r), 0)
    shift_c = - min(np.min(warped_c), 0)
    warped_r += shift_r
    warped_c += shift_c

    # Compute the dimensions of the transformed image
    warped_height = int(np.max(warped_r)) + 1
    warped_width = int(np.max(warped_c)) + 1

    output_quad_r, output_quad_c = polygon(warped_r[:4], warped_c[:4])

    img_warped = np.zeros((warped_height, warped_width, 4))

    img_warped[output_quad_r, output_quad_c] = interp2(warped_r, warped_c, output_quad_r, output_quad_c, img,
                                                       img_points)
    return img_warped, shift_r, shift_c


def warp2(img, correspondence):
    """
    Returns the warp of the image img to match the shape of
    the given correspondence matrix correspondence.
    """
    LOWER_BOUND_R = np.min(correspondence[3])
    UPPER_BOUND_R = np.max(correspondence[3])
    LOWER_BOUND_C = np.min(correspondence[4])
    UPPER_BOUND_C = np.max(correspondence[4])
    img_points = points(img)

    corr_matrix = computeH(correspondence)
    img_warped_points = np.matmul(corr_matrix, img_points)

    # transformed correspondence points
    warped_r = np.int0(img_warped_points[0, :] / img_warped_points[2, :])
    warped_c = np.int0(img_warped_points[1, :] / img_warped_points[2, :])

    cond = (warped_r > LOWER_BOUND_R) & (warped_r < UPPER_BOUND_R) & (warped_c > LOWER_BOUND_C) & (warped_c < UPPER_BOUND_C)
    warped_r = warped_r[cond]
    warped_c = warped_c[cond]
    img_points = img_points[:, cond]

    # Shift the transformed points so that they are all positive
    shift_r = -np.min(warped_r)  # - min(np.min(warped_r), 0)
    shift_c = -np.min(warped_c)  # - min(np.min(warped_c), 0)
    warped_r += shift_r
    warped_c += shift_c

    # (warped_r, warped_c) is img under affine transform
    # (output_quad_r, output_quad_c) determines the shape of interpolated region

    # Compute the dimensions of the transformed image
    warped_height = int(UPPER_BOUND_R - LOWER_BOUND_R + 1)
    warped_width = int(UPPER_BOUND_C - LOWER_BOUND_C + 1)

    img_warped = np.zeros((warped_height, warped_width, 4), dtype=np.float32)

    output_quad_r, output_quad_c = polygon(correspondence[3, :] + shift_r, correspondence[4, :] + shift_c)
    img_warped[output_quad_r, output_quad_c] = interp2(warped_r, warped_c, output_quad_r, output_quad_c, img, img_points)

    return img_warped
