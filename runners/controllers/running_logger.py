# python3.7
"""Contains the running controller to save the running log."""

import os
import json

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Ignore TF warning.

# pylint: disable=wrong-import-position
import torch
from torch.utils.tensorboard import SummaryWriter

from ..misc import format_time
from .base_controller import BaseController
# pylint: enable=wrong-import-position

__all__ = ['RunningLogger']


class RunningLogger(BaseController):
    """Defines the running controller to save the running log.

    This controller is able to save the log message in different formats:

    (1) Text format, which will be printed on screen and saved to the log file.
    (2) JSON format, which will be saved to `{runner.work_dir}/log.json`.
    (3) Tensorboard format.

    NOTE: The controller is set to `90` priority by default and will only be
    executed on the master worker.
    """

    def __init__(self, config=None):
        config = config or dict()
        config.setdefault('priority', 90)
        config.setdefault('every_n_iters', 1)
        config.setdefault('master_only', True)
        super().__init__(config)

        self._text_format = config.get('text_format', True)
        self._log_order = config.get('log_order', None)
        self._json_format = config.get('json_format', True)
        self._json_logpath = self._json_filename = 'log.json'
        self._tensorboard_format = config.get('tensorboard_format', True)
        self.tensorboard_writer = None

    def setup(self, runner):
        if self._text_format:
            runner.running_stats.log_order = self._log_order
        if self._json_format:
            self._json_logpath = os.path.join(
                runner.work_dir, self._json_filename)
        if self._tensorboard_format:
            event_dir = os.path.join(runner.work_dir, 'events')
            os.makedirs(event_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=event_dir)

    def close(self, runner):
        if self._tensorboard_format:
            self.tensorboard_writer.close()

    def execute_after_iteration(self, runner):
        # Prepare log data.
        log_data = {name: stats.get_log_value()
                    for name, stats in runner.running_stats.stats_pool.items()}

        # Save in text format.
        msg = f'Iter {runner.iter:6d}/{runner.total_iters:6d}'
        msg += f', {runner.running_stats}'
        memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        msg += f' (memory: {memory:.1f}G)'
        if 'iter_time' in log_data:
            eta = log_data['iter_time'] * (runner.total_iters - runner.iter)
            msg += f' (ETA: {format_time(eta)})'
        runner.logger.info(msg)

        # Save in JSON format.
        if self._json_format:
            with open(self._json_logpath, 'a+') as f:
                json.dump(log_data, f)
                f.write('\n')

        # Save in Tensorboard format.
        if self._tensorboard_format:
            for name, value in log_data.items():
                if name in ['data_time', 'iter_time', 'run_time']:
                    continue
                if name.startswith('loss_'):
                    self.tensorboard_writer.add_scalar(
                        name.replace('loss_', 'loss/'), value, runner.iter)
                elif name.startswith('lr_'):
                    self.tensorboard_writer.add_scalar(
                        name.replace('lr_', 'learning_rate/'), value, runner.iter)
                else:
                    self.tensorboard_writer.add_scalar(name, value, runner.iter)

        # Clear running stats.
        runner.running_stats.clear()
