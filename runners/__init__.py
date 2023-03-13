# python3.7
"""Collects all runners."""
from .stylegan_runner import StyleGANRunner,StyleGAN2Runner
from .encoder_runner import EncoderRunner

__all__ = ['StyleGANRunner', 'StyleGAN2Runner', 'EncoderRunner']
