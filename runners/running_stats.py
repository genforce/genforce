# python3.7
"""Contains the class for recording the running stats.

Here, running stats refers to the statictical information in the running
process, such as loss values, learning rates, running time, etc.
"""

from .misc import format_time

__all__ = ['SingleStats', 'RunningStats']


class SingleStats(object):
    """A class to record the stats corresponding to a particular variable.

    This class is log-friendly and supports customized log format, including:

    (1) Numerical log format, such as `.3f`, `.1e`, `05d`, and `>10s`.
    (2) Customized log name (name of the stats to show in the log).
    (3) Additional string (e.g., measure unit) as the tail of log message.

    Furthermore, this class also supports logging the stats with different
    strategies, including:

    (1) CURRENT: The current value will be logged.
    (2) AVERAGE: The averaged value (from the beginning) will be logged.
    (3) SUM: The cumulative value (from the beginning) will be logged.
    """

    def __init__(self,
                 name,
                 log_format='.3f',
                 log_name=None,
                 log_tail=None,
                 log_strategy='AVERAGE'):
        """Initializes the stats with log format.

        Args:
            name: Name of the stats. Should be a string without spaces.
            log_format: The numerical log format. Use `time` to log time
                duration. (default: `.3f`)
            log_name: The name shown in the log. `None` means to directly use
                the stats name. (default: None)
            log_tail: The tailing log message. (default: None)
            log_strategy: Strategy to log this stats. `CURRENT`, `AVERAGE`, and
                `SUM` are supported. (default: `AVERAGE`)

        Raises:
            ValueError: If the input `log_strategy` is not supported.
        """
        log_strategy = log_strategy.upper()
        if log_strategy not in ['CURRENT', 'AVERAGE', 'SUM']:
            raise ValueError(f'Invalid log strategy `{self.log_strategy}`!')

        self._name = name
        self._log_format = log_format
        self._log_name = log_name or name
        self._log_tail = log_tail or ''
        self._log_strategy = log_strategy

        # Stats Data.
        self.val = 0  # Current value.
        self.sum = 0  # Cumulative value.
        self.avg = 0  # Averaged value.
        self.cnt = 0  # Count number.

    @property
    def name(self):
        """Gets the name of the stats."""
        return self._name

    @property
    def log_format(self):
        """Gets tne numerical log format of the stats."""
        return self._log_format

    @property
    def log_name(self):
        """Gets the log name of the stats."""
        return self._log_name

    @property
    def log_tail(self):
        """Gets the tailing log message of the stats."""
        return self._log_tail

    @property
    def log_strategy(self):
        """Gets the log strategy of the stats."""
        return self._log_strategy

    def clear(self):
        """Clears the stats data."""
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.cnt = 0

    def update(self, value):
        """Updates the stats data."""
        self.val = value
        self.cnt = self.cnt + 1
        self.sum = self.sum + value
        self.avg = self.sum / self.cnt

    def get_log_value(self):
        """Gets value for logging according to the log strategy."""
        if self.log_strategy == 'CURRENT':
            return self.val
        if self.log_strategy == 'AVERAGE':
            return self.avg
        if self.log_strategy == 'SUM':
            return self.sum
        raise NotImplementedError(f'Log strategy `{self.log_strategy}` is not '
                                  f'implemented!')

    def __str__(self):
        """Gets log message."""
        if self.log_format == 'time':
            value_str = f'{format_time(self.get_log_value())}'
        else:
            value_str = f'{self.get_log_value():{self.log_format}}'
        return f'{self.log_name}: {value_str}{self.log_tail}'


class RunningStats(object):
    """A class to record all the running stats.

    Basically, this class contains a dictionary of SingleStats.

    Example:

    running_stats = RunningStats()
    running_stats.add('loss', log_format='.3f', log_strategy='AVERAGE')
    running_stats.add('time', log_format='time', log_name='Iter Time',
                      log_strategy='CURRENT')
    running_stats.log_order = ['time', 'loss']
    running_stats.update({'loss': 0.46, 'time': 12})
    running_stats.update({'time': 14.5, 'loss': 0.33})
    print(running_stats)
    """

    def __init__(self, log_delimiter=', '):
        """Initializes the running stats with the log delimiter.

        Args:
            log_delimiter: This delimiter is used to connect the log messages
                from different stats. (default: `, `)
        """
        self._log_delimiter = log_delimiter
        self.stats_pool = dict()  # The stats pool.
        self.log_order = None  # Order of the stats to log.

    @property
    def log_delimiter(self):
        """Gets the log delimiter between different stats."""
        return self._log_delimiter

    def add(self, name, **kwargs):
        """Adds a new SingleStats to the dictionary.

        Additional arguments include:

        log_format: The numerical log format. Use `time` to log time duration.
            (default: `.3f`)
        log_name: The name shown in the log. `None` means to directly use the
            stats name. (default: None)
        log_tail: The tailing log message. (default: None)
        log_strategy: Strategy to log this stats. `CURRENT`, `AVERAGE`, and
            `SUM` are supported. (default: `AVERAGE`)
        """
        if name in self.stats_pool:
            return
        self.stats_pool[name] = SingleStats(name, **kwargs)

    def clear(self, exclude_list=None):
        """Clears the stats data (if needed).

        Args:
            exclude_list: A list of stats names whose data will not be cleared.
        """
        exclude_list = set(exclude_list or [])
        for name, stats in self.stats_pool.items():
            if name not in exclude_list:
                stats.clear()

    def update(self, kwargs):
        """Updates the stats data by name."""
        for name, value in kwargs.items():
            if name not in self.stats_pool:
                self.add(name)
            self.stats_pool[name].update(value)

    def __getattr__(self, name):
        """Gets a particular SingleStats by name."""
        if name in self.stats_pool:
            return self.stats_pool[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f'`{self.__class__.__name__}` object has no '
                             f'attribute `{name}`!')

    def __str__(self):
        """Gets log message."""
        self.log_order = self.log_order or list(self.stats_pool)
        log_strings = [str(self.stats_pool[name]) for name in self.log_order]
        return self.log_delimiter.join(log_strings)
