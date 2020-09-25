# python3.7
"""Contains the running controller to handle checkpoints."""

import os.path

from .base_controller import BaseController

__all__ = ['Checkpointer']

class Checkpointer(BaseController):
    """Defines the running controller to handle checkpoints.

    This controller is used to save and load checkpoints.

    NOTE: This controller is set to `LAST` priority by default and will only be
    executed on the master worker.
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'LAST')
        config.setdefault('master_only', True)
        super().__init__(config)

        self._save_dir = config.get('checkpoint_dir', None)
        self._save_running_metadata = config.get('save_running_metadata', True)
        self._save_learning_rate = config.get('save_learning_rate', True)
        self._save_optimizer = config.get('save_optimizer', True)
        self._save_running_stats = config.get('save_running_stats', False)

    def execute_after_iteration(self, runner):
        save_dir = self._save_dir or runner.work_dir
        save_filename = f'checkpoint_iter{runner.iter:06d}.pth'
        runner.save(filepath=os.path.join(save_dir, save_filename),
                    running_metadata=self._save_running_metadata,
                    learning_rate=self._save_learning_rate,
                    optimizer=self._save_optimizer,
                    running_stats=self._save_running_stats)
