# python3.7
"""Contains the base class for model running controllers."""

__all__ = ['BaseController']

_CONTROLLER_PRIORITY_ALIASES = {
    'FIRST': 0,
    'HIGH': 25,
    'MEDIUM': 50,
    'LOW': 75,
    'LAST': 100,
}


def _parse_controller_priority(priority):
    """Parses the controller priority.

    Smaller number means higher priority. Controllers with higher priority will
    be executed first after the running iteration and executed last before the
    running iteration. Priority of controllers can be set in the configeration
    file. All priorities should be with integer type and lie in range [0, 100].

    Followings are some aliases for the default priorities:

    (1) FIRST: 0
    (2) HIGH: 25
    (3) MEDIUM: 50
    (4) LOW: 75
    (5) LAST: 100

    Args:
        priority: An integer or a string (alias) indicating the priority.

    Returns:
        An integer representing the parsed priority.

    Raises:
        TypeError: If the input `priority` is not with `int` or `str` type.
        ValueError: If the input `priority` is out of range [0, 100] or the
            `priority` is an invalid alias.
    """
    if isinstance(priority, int):
        if not 0 <= priority <= 100:
            raise ValueError(f'Controller priority should lie in range '
                             f'[0, 100], but `{priority}` is received!')
        return priority
    if isinstance(priority, str):
        try:
            return _CONTROLLER_PRIORITY_ALIASES[priority.upper()]
        except KeyError:
            raise ValueError(f'Unknown alias `{priority}` for controller '
                             f'priority!\n'
                             f'Please choose from: '
                             f'{list(_CONTROLLER_PRIORITY_ALIASES)}.')
    raise TypeError(f'Input `priority` should be with type `int` or `str`, '
                    f'but `{type(priority)}` is received!')


class BaseController(object):
    """The base class for model running controllers.

    Controllers are commonly used to control/monitor the running process, such
    as adjusting learning rate, saving log messages, etc. Within each iteration
    of model running, all controllers will be checked TWICE (i.e., before and
    after the iteration) on whether to execute the control.

    This class contains following members for a better control:

    (0) priority: Execution priority, which determines the execution order among
        all controllers. See function `_parse_controller_priority()` for more
        details. (default: `MEDIUM`)
    (1) every_n_iters: Executable for every n iterations. `-1` means ignored.
        (default: -1)
    (2) every_n_epochs: Executable for every n epochs. `-1` means ignored.
        (default: -1)
    (3) first_iter: Enforce to execute on the first iteration. (default: True)
    (4) marster_only: Executable only on the master worker. (default: False)
    """

    def __init__(self, config=None):
        """Initializes the controller with basic settings.

        Args:
            config: The configuration for the controller, which is loaded from
                the configuration file. This field should be a dictionary.
                (default: None)
        """
        config = config or dict()
        assert isinstance(config, dict)

        self._name = self.__class__.__name__
        self._config = config.copy()
        priority = config.get('priority', 'MEDIUM')
        self._priority = _parse_controller_priority(priority)
        self._every_n_iters = config.get('every_n_iters', -1)
        self._every_n_epochs = config.get('every_n_epochs', -1)
        self._first_iter = config.get('first_iter', True)
        self._master_only = config.get('master_only', False)

    @property
    def name(self):
        """Returns the name of the controller."""
        return self._name

    @property
    def config(self):
        """Returns the configuration for the controller."""
        return self._config

    @property
    def priority(self):
        """Returns the execution priority of the controller."""
        return self._priority

    @property
    def every_n_iters(self):
        """Returns how often (in iterations) the controller is executed."""
        return self._every_n_iters

    @property
    def every_n_epochs(self):
        """Returns how often (in epochs) the controller is executed."""
        return self._every_n_epochs

    @property
    def first_iter(self):
        """Returns whether the controller is forcibly executed initially."""
        return self._first_iter

    @property
    def master_only(self):
        """Returns whether the controller is executed on master worker only."""
        return self._master_only

    def setup(self, runner):
        """Sets up the controller before running.

        Default behavior is to do nothing. Can be overridden in derived classes.

        Args:
            runner: The runner to control.
        """

    def close(self, runner):
        """Closes the controller after running.

        Default behavior is to do nothing. Can be overridden in derived classes.

        Args:
            runner: The runner to control.
        """

    def execute_before_iteration(self, runner):
        """Executes the controller before the iteration.

        Default behavior is to do nothing. Can be overridden in derived classes.

        Args:
            runner: The runner to control.
        """

    def execute_after_iteration(self, runner):
        """Executes the controller after the iteration.

        Default behavior is to do nothing. Can be overridden in derived classes.

        Args:
            runner: The runner to control.
        """

    def is_executable(self, runner):
        """Determines whether the controller is executable at current state.

        Basically, the decision is made based on the current running iteration
        (epoch) and the execution frequency (i.e., `self.every_n_iters` and
        `self.every_n_epochs`).

        If `self.master_only` is set as `True`, this function will also check
        whether the current work is the master.

        Args:
            runner: The runner to control.

        Returns:
            A boolean suggesting whether the controller should be executed.
        """
        if self.master_only and runner.rank != 0:
            return False

        if self.first_iter and runner.iter - runner.start_iter == 1:
            return True
        if runner.iter == runner.total_iters:
            return True
        if self.every_n_iters > 0 and runner.iter % self.every_n_iters == 0:
            return True
        epoch_to_iter = runner.convert_epoch_to_iter(self.every_n_epochs)
        if self.every_n_epochs > 0 and runner.iter % epoch_to_iter == 0:
            return True
        return False

    def start(self, runner):
        """Starts the controller.

        Default behavior is to do nothing. Can be overridden in derived classes.

        Args:
            runner: The runner to control.
        """
        if self.master_only and runner.rank != 0:
            return
        self.setup(runner)

    def end(self, runner):
        """Ends the controller.

        Default behavior is to do nothing. Can be overridden in derived classes.

        Args:
            runner: The runner to control.
        """
        if self.master_only and runner.rank != 0:
            return
        self.close(runner)

    def pre_execute(self, runner):
        """Pre-executes the controller before the running of each iteration.

        This function wraps function `self.execute_before_iteration()` and
        function `self.is_executable()`. More concretely, the controller will
        only be executed at some particular iterations.

        Args:
            runner: The runner to control.
        """
        if self.is_executable(runner):
            self.execute_before_iteration(runner)

    def post_execute(self, runner):
        """Post-executes the controller after the running of each iteration.

        This function wraps function `self.execute_before_iteration()` and
        function `self.is_executable()`. More concretely, the controller will
        only be executed at some particular iterations.

        Args:
            runner: The runner to control.
        """
        if self.is_executable(runner):
            self.execute_after_iteration(runner)
