# python3.7
"""Collects all available models together."""

from .model_zoo import MODEL_ZOO
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .encoder import EncoderNet
from .perceptual_model import PerceptualModel

__all__ = [
    'MODEL_ZOO', 'PGGANGenerator', 'PGGANDiscriminator', 'StyleGANGenerator',
    'StyleGANDiscriminator', 'StyleGAN2Generator', 'StyleGAN2Discriminator',
    'EncoderNet', 'PerceptualModel', 'build_generator', 'build_discriminator',
    'build_encoder', 'build_perceptual', 'build_model'
]

_GAN_TYPES_ALLOWED = ['pggan', 'stylegan', 'stylegan2']
_MODULES_ALLOWED = ['generator', 'discriminator', 'encoder', 'perceptual']


def build_generator(gan_type, resolution, **kwargs):
    """Builds generator by GAN type.

    Args:
        gan_type: GAN type to which the generator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Generator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_discriminator(gan_type, resolution, **kwargs):
    """Builds discriminator by GAN type.

    Args:
        gan_type: GAN type to which the discriminator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Discriminator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_encoder(gan_type, resolution, **kwargs):
    """Builds encoder by GAN type.

    Args:
        gan_type: GAN type to which the encoder belong.
        resolution: Input resolution for encoder.
        **kwargs: Additional arguments to build the encoder.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type in ['stylegan', 'stylegan2']:
        return EncoderNet(resolution, **kwargs)

    raise NotImplementedError(f'Unsupported GAN type `{gan_type}` for encoder!')


def build_perceptual(**kwargs):
    """Builds perceptual model.

    Args:
        **kwargs: Additional arguments to build the encoder.
    """
    return PerceptualModel(**kwargs)


def build_model(gan_type, module, resolution, **kwargs):
    """Builds a GAN module (generator/discriminator/etc).

    Args:
        gan_type: GAN type to which the model belong.
        module: GAN module to build, such as generator or discrimiantor.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `module` is not supported.
        NotImplementedError: If the `module` is not implemented.
    """
    if module not in _MODULES_ALLOWED:
        raise ValueError(f'Invalid module: `{module}`!\n'
                         f'Modules allowed: {_MODULES_ALLOWED}.')

    if module == 'generator':
        return build_generator(gan_type, resolution, **kwargs)
    if module == 'discriminator':
        return build_discriminator(gan_type, resolution, **kwargs)
    if module == 'encoder':
        return build_encoder(gan_type, resolution, **kwargs)
    if module == 'perceptual':
        return build_perceptual(**kwargs)
    raise NotImplementedError(f'Unsupported module `{module}`!')
