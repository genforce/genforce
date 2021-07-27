# python3.7
"""Contains the base class for GAN runner."""

import os
import shutil
import numpy as np

import torch
import torch.distributed as dist

from metrics.inception import build_inception_model
from metrics.fid import extract_feature
from metrics.fid import compute_fid
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import postprocess_image
from utils.visualizer import save_image
from utils.visualizer import load_image
from .base_runner import BaseRunner

__all__ = ['BaseGANRunner']


class BaseGANRunner(BaseRunner):
    """Defines the base class for GAN runner."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.inception_model = None

    def moving_average_model(self, model, avg_model, beta=0.999):
        """Moving average model weights.

        This trick is commonly used in GAN training, where the weight of the
        generator is life-long averaged

        Args:
            model: The latest model used to update the averaged weights.
            avg_model: The averaged model weights.
            beta: Hyper-parameter used for moving average.
        """
        model_params = dict(self.get_module(model).named_parameters())
        avg_params = dict(self.get_module(avg_model).named_parameters())

        assert len(model_params) == len(avg_params)
        for param_name in avg_params:
            assert param_name in model_params
            avg_params[param_name].data = (
                avg_params[param_name].data * beta +
                model_params[param_name].data * (1 - beta))

    def build_models(self):
        super().build_models()
        assert 'generator' in self.models
        assert 'discriminator' in self.models
        self.z_space_dim = self.models['generator'].z_space_dim
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
                   z=None,
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

        temp_dir = os.path.join(self.work_dir, 'synthesize_results')
        os.makedirs(temp_dir, exist_ok=True)

        if z is not None:
            assert isinstance(z, np.ndarray)
            assert z.ndim == 2 and z.shape[1] == self.z_space_dim
            num = min(num, z.shape[0])
            z = torch.from_numpy(z).type(torch.FloatTensor)
        if not num:
            return
        # TODO: Use same z during the entire training process.

        self.logger.init_pbar()
        task1 = self.logger.add_pbar_task('Synthesize', total=num)

        indices = list(range(self.rank, num, self.world_size))
        for batch_idx in range(0, len(indices), self.val_batch_size):
            sub_indices = indices[batch_idx:batch_idx + self.val_batch_size]
            batch_size = len(sub_indices)
            if z is None:
                code = torch.randn(batch_size, self.z_space_dim).cuda()
            else:
                code = z[sub_indices].cuda()
            with torch.no_grad():
                if 'generator_smooth' in self.models:
                    G = self.models['generator_smooth']
                else:
                    G = self.models['generator']
                images = G(code, **self.G_kwargs_val)['image']
                images = postprocess_image(images.detach().cpu().numpy())
            for sub_idx, image in zip(sub_indices, images):
                save_image(os.path.join(temp_dir, f'{sub_idx:06d}.jpg'), image)
            self.logger.update_pbar(task1, batch_size * self.world_size)

        dist.barrier()
        if self.rank != 0:
            return

        if html_name:
            task2 = self.logger.add_pbar_task('Visualize', total=num)
            html = HtmlPageVisualizer(grid_size=num)
            for image_idx in range(num):
                image = load_image(
                    os.path.join(temp_dir, f'{image_idx:06d}.jpg'))
                row_idx, col_idx = divmod(image_idx, html.num_cols)
                html.set_cell(row_idx, col_idx, image=image,
                              text=f'Sample {image_idx:06d}')
                self.logger.update_pbar(task2, 1)
            html.save(os.path.join(self.work_dir, html_name))
        if not save_raw_synthesis:
            shutil.rmtree(temp_dir)

        self.logger.close_pbar()

    def fid(self,
            fid_num,
            z=None,
            ignore_cache=False,
            align_tf=True):
        """Computes the FID metric."""
        self.set_mode('val')

        if self.val_loader is None:
            self.build_dataset('val')
        fid_num = min(fid_num, len(self.val_loader.dataset))

        if self.inception_model is None:
            if align_tf:
                self.logger.info(f'Building inception model '
                                 f'(aligned with TensorFlow) ...')
            else:
                self.logger.info(f'Building inception model '
                                 f'(using torchvision) ...')
            self.inception_model = build_inception_model(align_tf).cuda()
            self.logger.info(f'Finish building inception model.')

        if z is not None:
            assert isinstance(z, np.ndarray)
            assert z.ndim == 2 and z.shape[1] == self.z_space_dim
            fid_num = min(fid_num, z.shape[0])
            z = torch.from_numpy(z).type(torch.FloatTensor)
        if not fid_num:
            return -1

        indices = list(range(self.rank, fid_num, self.world_size))

        self.logger.init_pbar()

        # Extract features from fake images.
        fake_feature_list = []
        task1 = self.logger.add_pbar_task('Fake', total=fid_num)
        for batch_idx in range(0, len(indices), self.val_batch_size):
            sub_indices = indices[batch_idx:batch_idx + self.val_batch_size]
            batch_size = len(sub_indices)
            if z is None:
                code = torch.randn(batch_size, self.z_space_dim).cuda()
            else:
                code = z[sub_indices].cuda()
            with torch.no_grad():
                if 'generator_smooth' in self.models:
                    G = self.models['generator_smooth']
                else:
                    G = self.models['generator']
                fake_images = G(code)['image']
                fake_feature_list.append(
                    extract_feature(self.inception_model, fake_images))
            self.logger.update_pbar(task1, batch_size * self.world_size)
        np.save(f'{self.work_dir}/fake_fid_features_{self.rank}.npy',
                np.concatenate(fake_feature_list, axis=0))

        # Extract features from real images if needed.
        cached_fid_file = f'{self.work_dir}/real_fid{fid_num}.npy'
        do_real_test = (not os.path.exists(cached_fid_file) or ignore_cache)
        if do_real_test:
            real_feature_list = []
            task2 = self.logger.add_pbar_task("Real", total=fid_num)
            for batch_idx in range(0, len(indices), self.val_batch_size):
                sub_indices = indices[batch_idx:batch_idx + self.val_batch_size]
                batch_size = len(sub_indices)
                data = next(self.val_loader)
                for key in data:
                    data[key] = data[key][:batch_size].cuda(
                        torch.cuda.current_device(), non_blocking=True)
                with torch.no_grad():
                    real_images = data['image']
                    real_feature_list.append(
                        extract_feature(self.inception_model, real_images))
                self.logger.update_pbar(task2, batch_size * self.world_size)
            np.save(f'{self.work_dir}/real_fid_features_{self.rank}.npy',
                    np.concatenate(real_feature_list, axis=0))

        dist.barrier()
        if self.rank != 0:
            return -1
        self.logger.close_pbar()

        # Collect fake features.
        fake_feature_list.clear()
        for rank in range(self.world_size):
            fake_feature_list.append(
                np.load(f'{self.work_dir}/fake_fid_features_{rank}.npy'))
            os.remove(f'{self.work_dir}/fake_fid_features_{rank}.npy')
        fake_features = np.concatenate(fake_feature_list, axis=0)
        assert fake_features.ndim == 2 and fake_features.shape[0] == fid_num
        feature_dim = fake_features.shape[1]
        pad = fid_num % self.world_size
        if pad:
            pad = self.world_size - pad
        fake_features = np.pad(fake_features, ((0, pad), (0, 0)))
        fake_features = fake_features.reshape(self.world_size, -1, feature_dim)
        fake_features = fake_features.transpose(1, 0, 2)
        fake_features = fake_features.reshape(-1, feature_dim)[:fid_num]

        # Collect (or load) real features.
        if do_real_test:
            real_feature_list.clear()
            for rank in range(self.world_size):
                real_feature_list.append(
                    np.load(f'{self.work_dir}/real_fid_features_{rank}.npy'))
                os.remove(f'{self.work_dir}/real_fid_features_{rank}.npy')
            real_features = np.concatenate(real_feature_list, axis=0)
            assert real_features.shape == (fid_num, feature_dim)
            real_features = np.pad(real_features, ((0, pad), (0, 0)))
            real_features = real_features.reshape(
                self.world_size, -1, feature_dim)
            real_features = real_features.transpose(1, 0, 2)
            real_features = real_features.reshape(-1, feature_dim)[:fid_num]
            np.save(cached_fid_file, real_features)
        else:
            real_features = np.load(cached_fid_file)
            assert real_features.shape == (fid_num, feature_dim)

        fid_value = compute_fid(fake_features, real_features)
        return fid_value
