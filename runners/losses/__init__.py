# python3.7
"""Collects all loss functions."""

from .logistic_gan_loss import LogisticGANLoss
from .encoder_loss import EncoderLoss

__all__ = ['LogisticGANLoss', 'EncoderLoss']
