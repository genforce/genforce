# python3.7
"""Contains the running controller for saving snapshot."""

from .base_controller import BaseController

__all__ = ['Snapshoter']


class Snapshoter(BaseController):
    """Defines the running controller for evaluation.

    NOTE: The controller is set to `LAST` priority by default.
    """

    def __init__(self, config):
        config.setdefault('priority', 'LAST')
        super().__init__(config)

        self.num = config.get('num', 100)

    def setup(self, runner):
        assert hasattr(runner, 'synthesize')

    def execute_after_iteration(self, runner):
        mode = runner.mode  # save runner mode.
        runner.synthesize(self.num,
                          html_name=f'snapshot_{runner.iter:06d}.html',
                          save_raw_synthesis=False)
        runner.logger.info(f'Saving snapshot at iter {runner.iter:06d} '
                           f'({runner.seen_img / 1000:.1f} kimg).')
        runner.set_mode(mode)  # restore runner mode.
