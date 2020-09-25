# python3.7
"""Contains the running controller to control progressive training.

This controller is applicable to the models that need to progressively change
the batch size, learning rate, etc.
"""

import numpy as np

from .base_controller import BaseController

__all__ = ['ProgressScheduler']

_BATCH_SIZE_SCHEDULE_DICT = {
    4: 16, 8: 8, 16: 4, 32: 2, 64: 1, 128: 1, 256: 1, 512: 1, 1024: 1,
}
_MAX_BATCH_SIZE = 64

_LEARNING_RATE_SCHEDULE_DICT = {
    4: 1, 8: 1, 16: 1, 32: 1, 64: 1, 128: 1.5, 256: 2, 512: 3, 1024: 3,
}


class ProgressScheduler(BaseController):
    """Defines the running controller to control progressive training.

    NOTE: The controller is set to `HIGH` priority by default.
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'HIGH')
        config.setdefault('every_n_iters', 1)
        super().__init__(config)

        self.base_batch_size = 0
        self.base_lrs = dict()

        self.total_img = 0
        self.init_res = config.get('init_res', 4)
        self.final_res = self.init_res
        self.init_lod = 0
        self.batch_size_schedule = config.get('batch_size_schedule', dict())
        self.lr_schedule = config.get('lr_schedule', dict())
        self.minibatch_repeats = config.get('minibatch_repeats', 4)

        self.lod_training_img = config.get('lod_training_img', 600_000)
        self.lod_transition_img = config.get('lod_transition_img', 600_000)
        self.lod_duration = (self.lod_training_img + self.lod_transition_img)

        # Whether to reset the optimizer state at the beginning of each phase.
        self.reset_optimizer = config.get('reset_optimizer', True)

    def get_batch_size(self, resolution):
        """Gets batch size for a particular resolution."""
        if self.batch_size_schedule:
            return self.batch_size_schedule.get(
                f'res{resolution}', self.base_batch_size)
        batch_size_scale = _BATCH_SIZE_SCHEDULE_DICT[resolution]
        return min(_MAX_BATCH_SIZE, self.base_batch_size * batch_size_scale)

    def get_lr_scale(self, resolution):
        """Gets learning rate scale for a particular resolution."""
        if self.lr_schedule:
            return self.lr_schedule.get(f'res{resolution}', 1)
        return _LEARNING_RATE_SCHEDULE_DICT[resolution]

    def setup(self, runner):
        # Set level of detail (lod).
        self.final_res = runner.resolution
        self.init_lod = np.log2(self.final_res // self.init_res)
        runner.lod = -1.0

        # Save default batch size and learning rate.
        self.base_batch_size = runner.batch_size
        for lr_name, lr_scheduler in runner.lr_schedulers.items():
            self.base_lrs[lr_name] = lr_scheduler.base_lrs

        # Add running stats for logging.
        runner.running_stats.add(
            'kimg', log_format='7.1f', log_name='kimg', log_strategy='CURRENT')
        runner.running_stats.add(
            'lod', log_format='4.2f', log_name='lod', log_strategy='CURRENT')
        runner.running_stats.add(
            'minibatch', log_format='4d', log_name='minibatch',
            log_strategy='CURRENT')

        # Log progressive schedule.
        runner.logger.info(f'Progressive Schedule:')
        res = self.init_res
        lod = int(self.init_lod)
        while res <= self.final_res:
            batch_size = self.get_batch_size(res)
            lr_scale = self.get_lr_scale(res)
            runner.logger.info(f'  Resolution {res:4d} (lod {lod}): '
                               f'batch size '
                               f'{batch_size:3d} * {runner.world_size:2d}, '
                               f'learning rate scale {lr_scale:.1f}')
            res *= 2
            lod -= 1
        assert lod == -1 and res == self.final_res * 2

        # Compute total running iterations.
        assert hasattr(runner.config, 'total_img')
        self.total_img = runner.config.total_img
        current_img = 0
        num_iters = 0
        while current_img < self.total_img:
            phase = (current_img + self.lod_transition_img) // self.lod_duration
            phase = np.clip(phase, 0, self.init_lod)
            if num_iters % self.minibatch_repeats == 0:
                resolution = self.init_res * (2 ** int(phase))
            current_img += self.get_batch_size(resolution) * runner.world_size
            num_iters += 1
        runner.total_iters = num_iters

    def execute_before_iteration(self, runner):
        is_first_iter = (runner.iter - runner.start_iter == 1)

        # Adjust hyper-parameters only at some particular iteration.
        if (not is_first_iter) and (runner.iter % self.minibatch_repeats != 1):
            return

        # Compute level-of-details.
        phase, subphase = divmod(runner.seen_img, self.lod_duration)
        lod = self.init_lod - phase
        if self.lod_transition_img:
            transition_img = max(subphase - self.lod_training_img, 0)
            lod = lod - transition_img / self.lod_transition_img
        lod = max(lod, 0.0)
        resolution = self.init_res * (2 ** int(np.ceil(self.init_lod - lod)))
        batch_size = self.get_batch_size(resolution)
        lr_scale = self.get_lr_scale(resolution)

        pre_lod = runner.lod
        pre_resolution = runner.train_loader.dataset.resolution
        runner.lod = lod

        # Reset optimizer state if needed.
        if self.reset_optimizer:
            if int(lod) != int(pre_lod) or np.ceil(lod) != np.ceil(pre_lod):
                runner.logger.info(f'Reset the optimizer state at '
                                   f'iter {runner.iter:06d} (lod {lod:.6f}).')
                for name in runner.optimizers:
                    runner.optimizers[name].state.clear()

        # Rebuild the dataset and adjust the learing rate if needed.
        if is_first_iter or resolution != pre_resolution:
            runner.logger.info(f'Rebuild the dataset at '
                               f'iter {runner.iter:06d} (lod {lod:.6f}).')
            runner.train_loader.overwrite_param(
                batch_size=batch_size, resolution=resolution)
            runner.batch_size = batch_size
            for lr_name, base_lrs in self.base_lrs.items():
                runner.lr_schedulers[lr_name].base_lrs = [
                    lr * lr_scale for lr in base_lrs]

    def execute_after_iteration(self, runner):
        minibatch = runner.batch_size * runner.world_size
        runner.running_stats.update({'kimg': runner.seen_img / 1000})
        runner.running_stats.update({'lod': runner.lod})
        runner.running_stats.update({'minibatch': minibatch})
