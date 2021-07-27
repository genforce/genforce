# python3.7
"""Defines loss functions for encoder training."""

import torch
import torch.nn.functional as F

from models import build_perceptual

__all__ = ['EncoderLoss']


class EncoderLoss(object):
    """Contains the class to compute logistic GAN loss."""

    def __init__(self,
                 runner,
                 d_loss_kwargs=None,
                 e_loss_kwargs=None,
                 perceptual_kwargs=None):
        """Initializes with models and arguments for computing losses."""
        self.d_loss_kwargs = d_loss_kwargs or dict()
        self.e_loss_kwargs = e_loss_kwargs or dict()
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        self.r2_gamma = self.d_loss_kwargs.get('r2_gamma', 0.0)

        self.perceptual_lw = self.e_loss_kwargs.get('perceptual_lw', 5e-5)
        self.adv_lw = self.e_loss_kwargs.get('adv_lw', 0.1)

        self.perceptual_model = build_perceptual(**perceptual_kwargs).cuda()
        self.perceptual_model.eval()
        for param in self.perceptual_model.parameters():
            param.requires_grad = False

        runner.space_of_latent = runner.config.space_of_latent

        runner.running_stats.add(
            f'recon_loss', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(
            f'adv_loss', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(
            f'loss_fake', log_format='.3f', log_strategy='AVERAGE')
        runner.running_stats.add(
            f'loss_real', log_format='.3f', log_strategy='AVERAGE')
        if self.r1_gamma != 0:
            runner.running_stats.add(
                f'real_grad_penalty', log_format='.3f', log_strategy='AVERAGE')
        if self.r2_gamma != 0:
            runner.running_stats.add(
                f'fake_grad_penalty', log_format='.3f', log_strategy='AVERAGE')

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
        if 'generator_smooth' in runner.models:
            G = runner.get_module(runner.models['generator_smooth'])
        else:
            G = runner.get_module(runner.models['generator'])
        G.eval()
        D = runner.models['discriminator']
        E = runner.models['encoder']

        reals = data['image']
        reals.requires_grad = True

        with torch.no_grad():
            latents = E(reals)
            if runner.space_of_latent == 'z':
                reals_rec = G(latents, **runner.G_kwargs_val)['image']
            elif runner.space_of_latent == 'wp':
                reals_rec = G.synthesis(latents,
                    **runner.G_kwargs_val)['image']
            elif runner.space_of_latent == 'y':
                G.set_space_of_latent('y')
                reals_rec = G.synthesis(latents,
                    **runner.G_kwargs_val)['image']
        real_scores = D(reals, **runner.D_kwargs_train)
        fake_scores = D(reals_rec, **runner.D_kwargs_train)
        loss_fake = F.softplus(fake_scores).mean()
        loss_real = F.softplus(-real_scores).mean()
        d_loss = loss_fake + loss_real

        runner.running_stats.update({'loss_fake': loss_fake.item()})
        runner.running_stats.update({'loss_real': loss_real.item()})

        real_grad_penalty = torch.zeros_like(d_loss)
        fake_grad_penalty = torch.zeros_like(d_loss)
        if self.r1_gamma:
            real_grad_penalty = self.compute_grad_penalty(reals, real_scores)
            runner.running_stats.update(
                {'real_grad_penalty': real_grad_penalty.item()})
        if self.r2_gamma:
            fake_grad_penalty = self.compute_grad_penalty(
                reals_rec, fake_scores)
            runner.running_stats.update(
                {'fake_grad_penalty': fake_grad_penalty.item()})

        return (d_loss +
                real_grad_penalty * (self.r1_gamma * 0.5) +
                fake_grad_penalty * (self.r2_gamma * 0.5))

    def e_loss(self, runner, data):
        """Computes loss for generator."""
        if 'generator_smooth' in runner.models:
            G = runner.get_module(runner.models['generator_smooth'])
        else:
            G = runner.get_module(runner.models['generator'])
        G.eval()
        D = runner.models['discriminator']
        E = runner.models['encoder']
        P = self.perceptual_model

        # Fetch data
        reals = data['image']

        latents = E(reals)
        if runner.space_of_latent == 'z':
            reals_rec = G(latents, **runner.G_kwargs_val)['image']
        elif runner.space_of_latent == 'wp':
            reals_rec = G.synthesis(latents, **runner.G_kwargs_val)['image']
        elif runner.space_of_latent == 'y':
            G.set_space_of_latent('y')
            reals_rec = G.synthesis(latents, **runner.G_kwargs_val)['image']
        loss_pix = F.mse_loss(reals_rec, reals, reduction='mean')
        loss_feat = self.perceptual_lw * F.mse_loss(
            P(reals_rec), P(reals), reduction='mean')
        loss_rec = loss_pix + loss_feat
        fake_scores = D(reals_rec, **runner.D_kwargs_train)
        adv_loss = self.adv_lw * F.softplus(-fake_scores).mean()
        e_loss = loss_pix + loss_feat + adv_loss

        runner.running_stats.update({'recon_loss': loss_rec.item()})
        runner.running_stats.update({'adv_loss': adv_loss.item()})

        return e_loss
