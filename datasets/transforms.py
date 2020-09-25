"""Contains transform functions."""

import cv2
import numpy as np
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'crop_resize_image', 'progressive_resize_image', 'resize_image',
    'normalize_image', 'normalize_latent_code', 'ImageResizing',
    'ImageNormalization', 'LatentCodeNormalization',
]


def crop_resize_image(image, size):
    """Crops a square patch and then resizes it to the given size.

    Args:
        image: The input image to crop and resize.
        size: An integer, indicating the target size.

    Returns:
        An image with target size.

    Raises:
        TypeError: If the input `image` is not with type `numpy.ndarray`.
        ValueError: If the input `image` is not with shape [H, W, C].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Input image should be with type `numpy.ndarray`, '
                        f'but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [H, W, C], '
                         f'but `{image.shape}` is received!')

    height, width, channel = image.shape
    short_side = min(height, width)
    image = image[(height - short_side) // 2:(height + short_side) // 2,
                  (width - short_side) // 2:(width + short_side) // 2]
    pil_image = PIL.Image.fromarray(image)
    pil_image = pil_image.resize((size, size), PIL.Image.ANTIALIAS)
    image = np.asarray(pil_image)
    assert image.shape == (size, size, channel)
    return image


def progressive_resize_image(image, size):
    """Resizes image to target size progressively.

    Different from normal resize, this function will reduce the image size
    progressively. In each step, the maximum reduce factor is 2.

    NOTE: This function can only handle square images, and can only be used for
    downsampling.

    Args:
        image: The input (square) image to resize.
        size: An integer, indicating the target size.

    Returns:
        An image with target size.

    Raises:
        TypeError: If the input `image` is not with type `numpy.ndarray`.
        ValueError: If the input `image` is not with shape [H, W, C].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Input image should be with type `numpy.ndarray`, '
                        f'but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [H, W, C], '
                         f'but `{image.shape}` is received!')

    height, width, channel = image.shape
    assert height == width
    assert height >= size
    num_iters = int(np.log2(height) - np.log2(size))
    for _ in range(num_iters):
        height = max(height // 2, size)
        image = cv2.resize(image, (height, height),
                           interpolation=cv2.INTER_LINEAR)
    assert image.shape == (size, size, channel)
    return image


def resize_image(image, size):
    """Resizes image to target size.

    NOTE: We use adaptive average pooing for image resizing. Instead of bilinear
    interpolation, average pooling is able to acquire information from more
    pixels, such that the resized results can be with higher quality.

    Args:
        image: The input image tensor, with shape [C, H, W], to resize.
        size: An integer or a tuple of integer, indicating the target size.

    Returns:
        An image tensor with target size.

    Raises:
        TypeError: If the input `image` is not with type `torch.Tensor`.
        ValueError: If the input `image` is not with shape [C, H, W].
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input image should be with type `torch.Tensor`, '
                        f'but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [C, H, W], '
                         f'but `{image.shape}` is received!')

    image = F.adaptive_avg_pool2d(image.unsqueeze(0), size).squeeze(0)
    return image


def normalize_image(image, mean=127.5, std=127.5):
    """Normalizes image by subtracting mean and dividing std.

    Args:
        image: The input image tensor to normalize.
        mean: The mean value to subtract from the input tensor. (default: 127.5)
        std: The standard deviation to normalize the input tensor. (default:
            127.5)

    Returns:
        A normalized image tensor.

    Raises:
        TypeError: If the input `image` is not with type `torch.Tensor`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input image should be with type `torch.Tensor`, '
                        f'but `{type(image)}` is received!')
    out = (image - mean) / std
    return out


def normalize_latent_code(latent_code, adjust_norm=True):
    """Normalizes latent code.

    NOTE: The latent code will always be normalized along the last axis.
    Meanwhile, if `adjust_norm` is set as `True`, the norm of the result will be
    adjusted to `sqrt(latent_code.shape[-1])` in order to avoid too small value.

    Args:
        latent_code: The input latent code tensor to normalize.
        adjust_norm: Whether to adjust the norm of the output. (default: True)

    Returns:
        A normalized latent code tensor.

    Raises:
        TypeError: If the input `latent_code` is not with type `torch.Tensor`.
    """
    if not isinstance(latent_code, torch.Tensor):
        raise TypeError(f'Input latent code should be with type '
                        f'`torch.Tensor`, but `{type(latent_code)}` is '
                        f'received!')
    dim = latent_code.shape[-1]
    norm = latent_code.pow(2).sum(-1, keepdim=True).pow(0.5)
    out = latent_code / norm
    if adjust_norm:
        out = out * (dim ** 0.5)
    return out


class ImageResizing(nn.Module):
    """Implements the image resizing layer."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, image):
        return resize_image(image, self.size)


class ImageNormalization(nn.Module):
    """Implements the image normalization layer."""

    def __init__(self, mean=127.5, std=127.5):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image):
        return normalize_image(image, self.mean, self.std)


class LatentCodeNormalization(nn.Module):
    """Implements the latent code normalization layer."""

    def __init__(self, adjust_norm=True):
        super().__init__()
        self.adjust_norm = adjust_norm

    def forward(self, latent_code):
        return normalize_latent_code(latent_code, self.adjust_norm)
