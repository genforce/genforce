# python3.7
"""Contains the base class for Encoder (GAN Inversion) runner."""

import os
import shutil

import torch
import torch.distributed as dist

from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import get_grid_shape
from utils.visualizer import postprocess_image
from utils.visualizer import save_image
from utils.visualizer import load_image
from .base_runner import BaseRunner

__all__ = ['BaseEncoderRunner']


class BaseEncoderRunner(BaseRunner):
    """Defines the base class for Encoder runner."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.inception_model = None

    def build_models(self):
        super().build_models()
        assert 'encoder' in self.models
        assert 'generator' in self.models
        assert 'discriminator' in self.models

        self.resolution = self.models['generator'].resolution
        self.G_kwargs_train = self.config.modules['generator'].get(
            'kwargs_train', dict())
        self.G_kwargs_val = self.config.modules['generator'].get(
            'kwargs_val', dict())
        self.D_kwargs_train = self.config.modules['discriminator'].get(
            'kwargs_train', dict())
        self.D_kwargs_val = self.config.modules['discriminator'].get(
            'kwargs_val', dict())

    def train_step(self, data, **train_kwargs):
        raise NotImplementedError('Should be implemented in derived class.')

    def val(self, **val_kwargs):
        self.synthesize(**val_kwargs)

    def synthesize(self,
                   num,
                   html_name=None,
                   save_raw_synthesis=False):
        """Synthesizes images.

        Args:
            num: Number of images to synthesize.
            z: Latent codes used for generation. If not specified, this function
                will sample latent codes randomly. (default: None)
            html_name: Name of the output html page for visualization. If not
                specified, no visualization page will be saved. (default: None)
            save_raw_synthesis: Whether to save raw synthesis on the disk.
                (default: False)
        """
        if not html_name and not save_raw_synthesis:
            return

        self.set_mode('val')

        if self.val_loader is None:
            self.build_dataset('val')

        temp_dir = os.path.join(self.work_dir, 'synthesize_results')
        os.makedirs(temp_dir, exist_ok=True)

        if not num:
            return
        if num % self.val_batch_size != 0:
            num =  (num //self.val_batch_size +1)*self.val_batch_size
        # TODO: Use same z during the entire training process.

        self.logger.init_pbar()
        task1 = self.logger.add_pbar_task('Synthesize', total=num)

        indices = list(range(self.rank, num, self.world_size))
        for batch_idx in range(0, len(indices), self.val_batch_size):
            sub_indices = indices[batch_idx:batch_idx + self.val_batch_size]
            batch_size = len(sub_indices)
            data = next(self.val_loader)
            for key in data:
                data[key] = data[key][:batch_size].cuda(
                    torch.cuda.current_device(), non_blocking=True)

            with torch.no_grad():
                real_images = data['image']
                E = self.models['encoder']
                if 'generator_smooth' in self.models:
                    G = self.get_module(self.models['generator_smooth'])
                else:
                    G = self.get_module(self.models['generator'])
                latents = E(real_images)
                if self.config.space_of_latent == 'z':
                    rec_images = G(
                        latents, **self.G_kwargs_val)['image']
                elif self.config.space_of_latent == 'wp':
                    rec_images = G.synthesis(
                        latents, **self.G_kwargs_val)['image']
                elif self.config.space_of_latent == 'y':
                    G.set_space_of_latent('y')
                    rec_images = G.synthesis(
                        latents, **self.G_kwargs_val)['image']
                else:
                    raise NotImplementedError(
                        f'Space of latent `{self.config.space_of_latent}` '
                        f'is not supported!')
                rec_images = postprocess_image(
                    rec_images.detach().cpu().numpy())
                real_images = postprocess_image(
                    real_images.detach().cpu().numpy())
            for sub_idx, rec_image, real_image in zip(
                    sub_indices, rec_images, real_images):
                save_image(os.path.join(temp_dir, f'{sub_idx:06d}_rec.jpg'),
                           rec_image)
                save_image(os.path.join(temp_dir, f'{sub_idx:06d}_ori.jpg'),
                           real_image)
            self.logger.update_pbar(task1, batch_size * self.world_size)

        dist.barrier()
        if self.rank != 0:
            return

        if html_name:
            task2 = self.logger.add_pbar_task('Visualize', total=num)
            row, col = get_grid_shape(num * 2)
            if row % 2 != 0:
                row, col = col, row
            html = HtmlPageVisualizer(num_rows=row, num_cols=col)
            for image_idx in range(num):
                rec_image = load_image(
                    os.path.join(temp_dir, f'{image_idx:06d}_rec.jpg'))
                real_image = load_image(
                    os.path.join(temp_dir, f'{image_idx:06d}_ori.jpg'))
                row_idx, col_idx = divmod(image_idx, html.num_cols)
                html.set_cell(2*row_idx, col_idx, image=real_image,
                              text=f'Sample {image_idx:06d}_ori')
                html.set_cell(2*row_idx+1, col_idx, image=rec_image,
                              text=f'Sample {image_idx:06d}_rec')
                self.logger.update_pbar(task2, 1)
            html.save(os.path.join(self.work_dir, html_name))
        if not save_raw_synthesis:
            shutil.rmtree(temp_dir)

        self.logger.close_pbar()
