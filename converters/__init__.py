# python3.7
"""Collects all model converters."""

from .pggan_converter import convert_pggan_weight
from .stylegan_converter import convert_stylegan_weight
from .stylegan2_converter import convert_stylegan2_weight
from .stylegan2ada_tf_converter import convert_stylegan2ada_tf_weight
from .stylegan2ada_pth_converter import convert_stylegan2ada_pth_weight

__all__ = [
    'convert_pggan_weight', 'convert_stylegan_weight',
    'convert_stylegan2_weight', 'convert_stylegan2ada_tf_weight',
    'convert_stylegan2ada_pth_weight'
]
