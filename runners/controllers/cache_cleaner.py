# python3.7
"""Contains the running controller to clean cache."""

import torch

from .base_controller import BaseController

__all__ = ['CacheCleaner']


class CacheCleaner(BaseController):
    """Defines the running controller to clean cache.

    This controller is used to empty the GPU cache after each iteration.

    NOTE: The controller is set to `LAST` priority by default.
    """

    def __init__(self, config=None):
        config = config or dict()
        config.setdefault('priority', 'LAST')
        config.setdefault('every_n_iters', 1)
        super().__init__(config)

    def setup(self, runner):
        torch.cuda.empty_cache()

    def close(self, runner):
        torch.cuda.empty_cache()

    def execute_after_iteration(self, runner):
        torch.cuda.empty_cache()
