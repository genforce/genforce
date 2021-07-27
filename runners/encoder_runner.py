# python3.7
"""Contains the runner for Encoder."""

from copy import deepcopy

from .base_encoder_runner import BaseEncoderRunner

__all__ = ['EncoderRunner']


class EncoderRunner(BaseEncoderRunner):
    """Defines the runner for Enccoder Training."""

    def build_models(self):
        super().build_models()
        if 'generator_smooth' not in self.models:
            self.models['generator_smooth'] = deepcopy(self.models['generator'])
            super().load(self.config.get('gan_model_path'),
                   running_metadata=False,
                   learning_rate=False,
                   optimizer=False,
                   running_stats=False)

    def train_step(self, data, **train_kwargs):
        self.set_model_requires_grad('generator', False)

        # E_loss
        self.set_model_requires_grad('discriminator', False)
        self.set_model_requires_grad('encoder', True)
        E_loss = self.loss.e_loss(self, data)
        self.optimizers['encoder'].zero_grad()
        E_loss.backward()
        self.optimizers['encoder'].step()

        # D_loss
        self.set_model_requires_grad('discriminator', True)
        self.set_model_requires_grad('encoder', False)
        D_loss = self.loss.d_loss(self, data)
        self.optimizers['discriminator'].zero_grad()
        D_loss.backward()
        self.optimizers['discriminator'].step()

    def load(self, **kwargs):
        super().load(**kwargs)
