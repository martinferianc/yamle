# -*- coding: utf-8 -*-
"""
This file contains code for the ImageNet-C corruptions, adapted from the original code available at https://github.com/hendrycks/robustness. The adaptation includes:

- Variable image size (not fixed to 256x256).
- Removal of FSGM (Fast Sign Gradient Method).
- Changes to the Motion blur implementation, avoiding the use of the Wand library.
"""

import numpy as np

# /////////////// Corruption Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates, rotate
from pkg_resources import resource_filename


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):  # TODO: needs img size configuration
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=1
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top : trim_top + h, trim_top : trim_top + h]


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions ///////////////


def gaussian_noise(x, img_size: int, severity=1):
    if img_size == 64:
        c = [0.04, 0.08, 0.12, 0.15, 0.18][severity - 1]
    elif img_size <= 32:
        c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    elif img_size >= 65:
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    else:
        raise ValueError(f"Gaussian noise not defined for image size {img_size}")

    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, img_size: int, severity=1):
    if img_size == 64:
        c = [250, 100, 50, 30, 15][severity - 1]
    elif img_size <= 32:
        c = [500, 250, 100, 75, 50][severity - 1]
    elif img_size >= 65:
        c = [60, 25, 12, 5, 3][severity - 1]
    else:
        raise ValueError(f"Shot noise not defined for image size {img_size}")

    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, img_size: int, severity=1):
    if img_size == 64:
        c = [0.01, 0.02, 0.05, 0.08, 0.14][severity - 1]
    elif img_size <= 32:
        c = [0.01, 0.02, 0.03, 0.05, 0.07][severity - 1]
    elif img_size >= 65:
        c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    else:
        raise ValueError(f"Impulse noise not defined for image size {img_size}")

    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, img_size: int, severity=1):
    if img_size == 64:
        c = [0.15, 0.2, 0.25, 0.3, 0.35][severity - 1]
    elif img_size <= 32:
        c = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]
    elif img_size >= 65:
        c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
    else:
        raise ValueError(f"Speckle noise not defined for image size {img_size}")

    x = np.array(x) / 255.0
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, img_size: int, severity=1):
    if img_size == 64:
        c = [0.5, 0.75, 1, 1.25, 1.5][severity - 1]
    elif img_size <= 32:
        c = [0.4, 0.6, 0.7, 0.8, 1][severity - 1]
    elif img_size >= 65:
        c = [1, 2, 3, 4, 6][severity - 1]
    else:
        raise ValueError(f"Gaussian blur not defined for image size {img_size}")

    x = gaussian(np.array(x) / 255.0, sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, img_size: int, severity=1):
    # sigma, max_delta, iterations
    if img_size == 64:
        c = [(0.1, 1, 1), (0.5, 1, 1), (0.6, 1, 2), (0.7, 2, 1), (0.9, 2, 2)][
            severity - 1
        ]
    elif img_size <= 32:
        c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][
            severity - 1
        ]
    elif img_size >= 65:
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][
            severity - 1
        ]
    else:
        raise ValueError(f"Glass blur not defined for image size {img_size}")

    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis=-1) * 255)

    # locally shuffle pixels
    """
    for i in range(c[2]):
        for h in range(img_size - c[1], c[1], -1):
            for w in range(img_size - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    """
    # Write the above loop in a faster way
    for i in range(c[2]):
        h = np.arange(img_size - c[1], c[1], -1)
        w = np.arange(img_size - c[1], c[1], -1)
        # Create a meshgrid
        h, w = np.meshgrid(h, w)
        # Create a random offset
        dx, dy = np.random.randint(
            -c[1], c[1], size=(2, img_size - 2 * c[1], img_size - 2 * c[1])
        )
        # Add the offset to the meshgrid
        h_prime, w_prime = h + dy, w + dx
        # Swap the pixels
        x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis=-1), 0, 1) * 255


def defocus_blur(x, img_size: int, severity=1):
    if img_size == 64:
        c = [(0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1)][severity - 1]
    elif img_size <= 32:
        c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    elif img_size >= 65:
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    else:
        raise ValueError(f"Defocus blur not defined for image size {img_size}")

    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def _motion_blur(x, kernel_size, sigma, angle):
    x = np.array(x) / 255.0
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    # Change it to a horizontal kernel
    horizontal_kernel = np.zeros((kernel_size, kernel_size))
    horizontal_kernel[:, int((kernel_size - 1) / 2)] = gaussian_kernel[:, 0]
    # Rotate the kernel with the given angle
    kernel = rotate(horizontal_kernel, angle)
    channels = []
    for d in range(x.shape[2]):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
    return np.clip(channels, 0, 1) * 255


def motion_blur(x, img_size: int, severity=1):
    if img_size == 64:
        c = [(10, 1), (10, 1.5), (10, 2), (10, 2.5), (12, 3)][severity - 1]
    elif img_size <= 32:
        c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]
    elif img_size >= 65:
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    else:
        raise ValueError(f"Motion blur not defined for image size {img_size}")
    kernel_size = c[0] * 2 + 1
    sigma = c[1]
    angle = np.random.uniform(-45, 45)
    x = _motion_blur(x, kernel_size, sigma, angle)
    return x


def zoom_blur(x, img_size: int, severity=1):
    if img_size == 64:
        c = [
            np.arange(1, 1.06, 0.01),
            np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.01),
            np.arange(1, 1.26, 0.01),
        ][severity - 1]
    elif img_size <= 32:
        c = [
            np.arange(1, 1.06, 0.01),
            np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.01),
            np.arange(1, 1.26, 0.01),
        ][severity - 1]
    elif img_size >= 65:
        c = [
            np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.02),
            np.arange(1, 1.26, 0.02),
            np.arange(1, 1.31, 0.03),
        ][severity - 1]
    else:
        raise ValueError(f"Zoom blur not defined for image size {img_size}")

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, img_size: int, severity=1):
    if img_size == 64:
        c = [(0.4, 3), (0.7, 3), (1, 2.5), (1.5, 2), (2, 1.75)][severity - 1]
    elif img_size <= 32:
        c = [(0.2, 3), (0.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    elif img_size >= 65:
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    else:
        raise ValueError(f"Fog not defined for image size {img_size}")

    x = np.array(x) / 255.0
    max_val = x.max()
    x += (
        c[0]
        * plasma_fractal(mapsize=img_size, wibbledecay=c[1])[:img_size, :img_size][
            ..., np.newaxis
        ]
    )
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, img_size: int, severity=1):
    if img_size == 64:
        c = [(1, 0.3), (0.9, 0.4), (0.8, 0.45), (0.75, 0.5), (0.7, 0.55)][severity - 1]
    elif img_size <= 32:
        c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
    elif img_size >= 65:
        c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    else:
        raise ValueError(f"Frost not defined for image size {img_size}")
    idx = np.random.randint(5)
    filename = [
        resource_filename(__name__, "frost/frost1.png"),
        resource_filename(__name__, "frost/frost2.png"),
        resource_filename(__name__, "frost/frost3.png"),
        resource_filename(__name__, "frost/frost4.jpg"),
        resource_filename(__name__, "frost/frost5.jpg"),
        resource_filename(__name__, "frost/frost6.jpg"),
    ][idx]
    frost = cv2.imread(filename)
    if img_size == 64:
        frost = cv2.resize(frost, (0, 0), fx=0.3, fy=0.3)
    elif img_size <= 32:
        frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
    else:
        raise ValueError(f"Frost not defined for image size {img_size}")
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(
        0, frost.shape[0] - img_size
    ), np.random.randint(0, frost.shape[1] - img_size)
    frost = frost[x_start : x_start + img_size, y_start : y_start + img_size][
        ..., [2, 1, 0]
    ]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, img_size: int, severity=1):
    if img_size == 64:
        c = [
            (0.1, 0.2, 1, 0.6, 8, 3, 0.8),
            (0.1, 0.2, 1, 0.5, 10, 4, 0.8),
            (0.15, 0.3, 1.75, 0.55, 10, 4, 0.7),
            (0.25, 0.3, 2.25, 0.6, 12, 6, 0.65),
            (0.3, 0.3, 1.25, 0.65, 14, 12, 0.6),
        ][severity - 1]
    elif img_size <= 32:
        c = [
            (0.1, 0.2, 1, 0.6, 8, 3, 0.95),
            (0.1, 0.2, 1, 0.5, 10, 4, 0.9),
            (0.15, 0.3, 1.75, 0.55, 10, 4, 0.9),
            (0.25, 0.3, 2.25, 0.6, 12, 6, 0.85),
            (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8),
        ][severity - 1]
    elif img_size >= 65:
        c = [
            (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
        ][severity - 1]
    else:
        raise ValueError(f"Snow not defined for image size {img_size}")

    x = np.array(x, dtype=np.float32) / 255.0
    snow_layer = np.random.normal(
        size=x.shape[:2], loc=c[0], scale=c[1]
    )  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = np.clip(snow_layer, 0, 1) * 255
    kernel_size = c[4] * 2 + 1
    sigma = c[5]
    angle = np.random.uniform(-135, -45)
    snow_layer = _motion_blur(snow_layer, kernel_size, sigma, angle) / 255.0

    x = c[6] * x + (1 - c[6]) * np.maximum(
        x,
        cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(img_size, img_size, 1) * 1.5 + 0.5,
    )
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, img_size: int, severity=1):
    if img_size == 64:
        c = [
            (0.62, 0.1, 0.7, 0.7, 0.6, 0),
            (0.65, 0.1, 0.8, 0.7, 0.6, 0),
            (0.65, 0.3, 1, 0.69, 0.6, 0),
            (0.65, 0.1, 0.7, 0.68, 0.6, 1),
            (0.65, 0.1, 0.5, 0.67, 0.6, 1),
        ][severity - 1]
    elif img_size <= 32:
        c = [
            (0.62, 0.1, 0.7, 0.7, 0.5, 0),
            (0.65, 0.1, 0.8, 0.7, 0.5, 0),
            (0.65, 0.3, 1, 0.69, 0.5, 0),
            (0.65, 0.1, 0.7, 0.69, 0.6, 1),
            (0.65, 0.1, 0.5, 0.68, 0.6, 1),
        ][severity - 1]
    elif img_size >= 65:
        c = [
            (0.65, 0.3, 4, 0.69, 0.6, 0),
            (0.65, 0.3, 3, 0.68, 0.6, 0),
            (0.65, 0.3, 2, 0.68, 0.5, 0),
            (0.65, 0.3, 1, 0.65, 1.5, 1),
            (0.67, 0.4, 1, 0.65, 1.5, 1),
        ][severity - 1]
    else:
        raise ValueError(f"Spatter not defined for image size {img_size}")

    x = np.array(x, dtype=np.float32) / 255.0

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate(
            (
                175 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
            ),
            axis=2,
        )

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate(
            (
                63 / 255.0 * np.ones_like(x[..., :1]),
                42 / 255.0 * np.ones_like(x[..., :1]),
                20 / 255.0 * np.ones_like(x[..., :1]),
            ),
            axis=2,
        )

        color *= m[..., np.newaxis]
        x *= 1 - m[..., np.newaxis]

        return np.clip(x + color, 0, 1) * 255


def contrast(x, img_size: int, severity=1):
    if img_size == 64:
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    elif img_size <= 32:
        c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]
    elif img_size >= 65:
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    else:
        raise ValueError(f"Contrast not defined for image size {img_size}")

    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, img_size: int, severity=1):
    if img_size == 64:
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    elif img_size <= 32:
        c = [0.05, 0.1, 0.15, 0.2, 0.3][severity - 1]
    elif img_size >= 65:
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    else:
        raise ValueError(f"Brightness not defined for image size {img_size}")

    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, img_size: int, severity=1):
    if img_size == 64:
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (30, 0.2)][severity - 1]
    elif img_size <= 32:
        c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    elif img_size >= 65:
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    else:
        raise ValueError(f"Saturate not defined for image size {img_size}")

    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, img_size: int, severity=1):
    if img_size == 64:
        c = [65, 58, 50, 40, 25][severity - 1]
    elif img_size <= 32:
        c = [80, 65, 58, 50, 40][severity - 1]
    elif img_size >= 65:
        c = [25, 18, 15, 10, 7][severity - 1]
    else:
        raise ValueError(f"JPEG compression not defined for image size {img_size}")

    output = BytesIO()
    x.save(output, "JPEG", quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, img_size: int, severity=1):
    if img_size == 64:
        c = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]
    elif img_size <= 32:
        c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    elif img_size >= 65:
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    else:
        raise ValueError(f"Pixelate not defined for image size {img_size}")

    x = x.resize((int(img_size * c), int(img_size * c)), PILImage.BOX)
    x = x.resize((img_size, img_size), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(x, img_size: int, severity=1):
    if img_size >= 32 and img_size <= 64:
        c = [
            (img_size * 0, img_size * 0, img_size * 0.08),
            (img_size * 0.05, img_size * 0.2, img_size * 0.07),
            (img_size * 0.08, img_size * 0.06, img_size * 0.06),
            (img_size * 0.1, img_size * 0.04, img_size * 0.05),
            (img_size * 0.1, img_size * 0.03, img_size * 0.03),
        ][severity - 1]
    elif img_size >= 65:
        c = [
            (img_size * 2, img_size * 0.7, img_size * 0.1),
            (img_size * 2, img_size * 0.08, img_size * 0.2),
            (img_size * 0.05, img_size * 0.01, img_size * 0.02),
            (img_size * 0.07, img_size * 0.01, img_size * 0.02),
            (img_size * 0.12, img_size * 0.01, img_size * 0.02),
        ][severity - 1]
    else:
        raise ValueError(f"Elastic transform not defined for image size {img_size}")

    image = x
    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dy = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return (
        np.clip(
            map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
            0,
            1,
        )
        * 255
    )


# /////////////// End Corruptions ///////////////
