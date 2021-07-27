# python3.7
"""Defines loss functions for GAN training."""

import numpy as np

import torch
import torch.nn.functional as F

__all__ = ['LogisticGANLoss']

apply_loss_scaling = lambda x: x * torch.exp(x * np.log(2.0))
undo_loss_scaling = lambda x: x * torch.exp(-x * np.log(2.0))


class LogisticGANLoss(object):
    """Contains the class to compute logistic GAN loss."""

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""
        self.d_loss_kwargs = d_loss_kwargs or dict()
        self.g_loss_kwargs = g_loss_kwargs or dict()
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        self.r2_gamma = self.d_loss_kwargs.get('r2_gamma', 0.0)

        runner.running_stats.add(
            f'g_loss', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(
            f'd_loss', log_format='.3f', log_strategy='AVERAGE')
        if self.r1_gamma != 0:
            runner.running_stats.add(
                f'real_grad_penalty', log_format='.3f', log_strategy='AVERAGE')
        if self.r2_gamma != 0:
            runner.running_stats.add(
                f'fake_grad_penalty', log_format='.3f', log_strategy='AVERAGE')

    @staticmethod
    def preprocess_image(images, lod=0, **_unused_kwargs):
        """Pre-process images."""
        if lod != int(lod):
            downsampled_images = F.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
            upsampled_images = F.interpolate(
                downsampled_images, scale_factor=2, mode='nearest')
            alpha = lod - int(lod)
            images = images * (1 - alpha) + upsampled_images * alpha
        if int(lod) == 0:
            return images
        return F.interpolate(
            images, scale_factor=(2 ** int(lod)), mode='nearest')

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        image_grad = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=images,
            create_graph=True,
            retain_graph=True)[0].view(images.shape[0], -1)
        penalty = image_grad.pow(2).sum(dim=1).mean()
        return penalty

    def d_loss(self, runner, data):
        """Computes loss for discriminator."""
        G = runner.models['generator']
        D = runner.models['discriminator']

        reals = self.preprocess_image(data['image'], lod=runner.lod)
        reals.requires_grad = True
        labels = data.get('label', None)

        latents = torch.randn(reals.shape[0], runner.z_space_dim).cuda()
        latents.requires_grad = True
        # TODO: Use random labels.
        fakes = G(latents, label=labels, **runner.G_kwargs_train)['image']
        real_scores = D(reals, label=labels, **runner.D_kwargs_train)
        fake_scores = D(fakes, label=labels, **runner.D_kwargs_train)

        d_loss = F.softplus(fake_scores).mean()
        d_loss += F.softplus(-real_scores).mean()
        runner.running_stats.update({'d_loss': d_loss.item()})

        real_grad_penalty = torch.zeros_like(d_loss)
        fake_grad_penalty = torch.zeros_like(d_loss)
        if self.r1_gamma:
            real_grad_penalty = self.compute_grad_penalty(reals, real_scores)
            runner.running_stats.update(
                {'real_grad_penalty': real_grad_penalty.item()})
        if self.r2_gamma:
            fake_grad_penalty = self.compute_grad_penalty(fakes, fake_scores)
            runner.running_stats.update(
                {'fake_grad_penalty': fake_grad_penalty.item()})

        return (d_loss +
                real_grad_penalty * (self.r1_gamma * 0.5) +
                fake_grad_penalty * (self.r2_gamma * 0.5))

    def g_loss(self, runner, data):  # pylint: disable=no-self-use
        """Computes loss for generator."""
        # TODO: Use random labels.
        G = runner.models['generator']
        D = runner.models['discriminator']
        batch_size = data['image'].shape[0]
        labels = data.get('label', None)

        latents = torch.randn(batch_size, runner.z_space_dim).cuda()
        fakes = G(latents, label=labels, **runner.G_kwargs_train)['image']
        fake_scores = D(fakes, label=labels, **runner.D_kwargs_train)

        g_loss = F.softplus(-fake_scores).mean()
        runner.running_stats.update({'g_loss': g_loss.item()})

        return g_loss
