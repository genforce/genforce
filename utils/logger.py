# python3.7
"""Utility functions for logging."""

import os
import sys
import logging
from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import TextColumn
from rich.progress import BarColumn
from rich.text import Text

__all__ = ['build_logger', 'Logger', 'RichLogger', 'DumbLogger']

DEFAULT_WORK_DIR = 'work_dirs'

_LOGGER_TYPES = ['normal', 'rich', 'dumb']


def build_logger(logger_type='normal', **kwargs):
    """Builds a logger.

    Supported Logger types:
        `normal`: The default logger.
        `rich`: Record messages with decoration, using `rich` module.
        `dumb`: Do NOT record any message.

    Args:
        logger_type: Type of logger, which is case insensitive.
            (default: `normal`)
        **kwargs: Additional arguments.
    """
    assert isinstance(logger_type, str)
    logger_type = logger_type.lower()
    if logger_type not in _LOGGER_TYPES:
        raise ValueError(f'Invalid logger type `{logger_type}`!\n'
                         f'Types allowed: {_LOGGER_TYPES}.')
    if logger_type == 'normal':
        return Logger(**kwargs)
    if logger_type == 'rich':
        return RichLogger(**kwargs)
    if logger_type == 'dumb':
        return DumbLogger(**kwargs)
    raise NotImplementedError(f'Not implemented logger type `{logger_type}`!')


class Logger(object):
    """Defines a logger to record log message both on screen and to file.

    The class sets up a logger with `DEBUG` log level. Two handlers will be
    added to the logger. One is the `sys.stderr` stream, with `INFO` log level,
    which will print improtant messages on the screen. The other is used to save
    all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will be added time
    stamp and log level before logged.

    NOTE: If `logfile_name` is empty, the file stream will be skipped.
    """

    def __init__(self,
                 work_dir=DEFAULT_WORK_DIR,
                 logfile_name='log.txt',
                 logger_name='logger'):
        """Initializes the logger.

        Args:
            work_dir: The work directory. (default: DEFAULT_WORK_DIR)
            logfile_name: Name of the log file. (default: `log.txt`)
            logger_name: Unique name for the logger. (default: `logger`)
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        if self.logger.hasHandlers():  # Already existed
            raise SystemExit(f'Logger `{logger_name}` has already existed!\n'
                             f'Please use another name, or otherwise the '
                             f'messages from these two logger may be mixed up.')

        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # Print log message with `INFO` level or above onto the screen.
        terminal_handler = logging.StreamHandler(stream=sys.stdout)
        terminal_handler.setLevel(logging.INFO)
        terminal_handler.setFormatter(formatter)
        self.logger.addHandler(terminal_handler)

        # Save log message with all levels into log file if needed.
        if logfile_name:
            os.makedirs(work_dir, exist_ok=True)
            self.file_stream = open(os.path.join(work_dir, logfile_name), 'a')
            file_handler = logging.StreamHandler(stream=self.file_stream)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.log = self.logger.log
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.exception = self.logger.exception
        self.critical = self.logger.critical

        self.pbar = []
        self.pbar_kwargs = None

    def __del__(self):
        if hasattr(self, 'file_stream'):
            self.file_stream.close()

    def print(self, *messages, **_unused_kwargs):
        """Prints messages without time stamp or log level."""
        for handler in self.logger.handlers:
            print(*messages, file=handler.stream)

    def init_pbar(self, leave=False):
        """Initializes a progress bar which will display on the screen only.

        Args:
            leave: Whether to leave the trace. (default: False)
        """
        columns = [
            '{desc}',
            '{bar}',
            ' {percentage:5.1f}%',
            '[{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        ]
        self.pbar_kwargs = dict(
            leave=leave,
            bar_format=' '.join(columns),
            unit='',
        )

    def add_pbar_task(self, name, total):
        """Adds a task to the progress bar.

        Args:
            name: Name of the new task.
            total: Total number of steps (samples) contained in the task.

        Returns:
            The task ID.
        """
        assert isinstance(self.pbar_kwargs, dict)
        self.pbar.append(tqdm(desc=name, total=total, **self.pbar_kwargs))
        return len(self.pbar) - 1

    def update_pbar(self, task_id, advance=1):
        """Updates a certain task in the progress bar.

        Args:
            task_id: ID of the task to update.
            advance: Number of steps advanced onto the target task. (default: 1)
        """
        assert len(self.pbar) > task_id and isinstance(self.pbar[task_id], tqdm)
        if self.pbar[task_id].n < self.pbar[task_id].total:
            self.pbar[task_id].update(advance)
            if self.pbar[task_id].n >= self.pbar[task_id].total:
                self.pbar[task_id].refresh()

    def close_pbar(self):
        """Closes the progress bar"""
        for pbar in self.pbar[::-1]:
            pbar.close()
        self.pbar.clear()
        self.pbar_kwargs = None


def _format_time(seconds):
    """Formats seconds to readable time string.

    This function is used to display time in progress bar.
    """
    if not seconds:
        return '--:--'

    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


class TimeColumn(ProgressColumn):
    """Renders total time, ETA, and speed in progress bar."""

    max_refresh = 0.5  # Only refresh twice a second to prevent jitter

    def render(self, task):
        elapsed_time = _format_time(task.elapsed)
        eta = _format_time(task.time_remaining)
        speed = f'{task.speed:.2f}/s' if task.speed else '?/s'
        return Text(f'[{elapsed_time}<{eta}, {speed}]',
                    style="progress.remaining")


class RichLogger(object):
    """Defines a logger based on `rich.RichHandler`.

    Compared to the basic Logger, this logger will decorate the log message in
    a pretty format automatically.
    """

    def __init__(self,
                 work_dir=DEFAULT_WORK_DIR,
                 logfile_name='log.txt',
                 logger_name='logger'):
        """Initializes the logger.

        Args:
            work_dir: The work directory. (default: DEFAULT_WORK_DIR)
            logfile_name: Name of the log file. (default: `log.txt`)
            logger_name: Unique name for the logger. (default: `logger`)
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        if self.logger.hasHandlers():  # Already existed
            raise SystemExit(f'Logger `{logger_name}` has already existed!\n'
                             f'Please use another name, or otherwise the '
                             f'messages from these two logger may be mixed up.')

        self.logger.setLevel(logging.DEBUG)

        # Print log message with `INFO` level or above onto the screen.
        terminal_console = Console(
            file=sys.stderr, log_time=False, log_path=False)
        terminal_handler = RichHandler(
            level=logging.INFO,
            console=terminal_console,
            show_time=True,
            show_level=True,
            show_path=False)
        terminal_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(terminal_handler)

        # Save log message with all levels into log file if needed.
        if logfile_name:
            os.makedirs(work_dir, exist_ok=True)
            self.file_stream = open(os.path.join(work_dir, logfile_name), 'a')
            file_console = Console(
                file=self.file_stream, log_time=False, log_path=False)
            file_handler = RichHandler(
                level=logging.DEBUG,
                console=file_console,
                show_time=True,
                show_level=True,
                show_path=False)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)

        self.log = self.logger.log
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.exception = self.logger.exception
        self.critical = self.logger.critical

        self.pbar = None

    def __del__(self):
        if hasattr(self, 'file_stream'):
            self.file_stream.close()

    def print(self, *messages, **kwargs):
        """Prints messages without time stamp or log level."""
        for handler in self.logger.handlers:
            handler.console.print(*messages, **kwargs)

    def init_pbar(self, leave=False):
        """Initializes a progress bar which will display on the screen only.

        Args:
            leave: Whether to leave the trace. (default: False)
        """
        assert self.pbar is None

        # Columns shown in the progress bar.
        columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
            TimeColumn(),
        )

        self.pbar = Progress(*columns,
                             console=self.logger.handlers[0].console,
                             transient=not leave,
                             auto_refresh=True,
                             refresh_per_second=10)
        self.pbar.start()

    def add_pbar_task(self, name, total):
        """Adds a task to the progress bar.

        Args:
            name: Name of the new task.
            total: Total number of steps (samples) contained in the task.

        Returns:
            The task ID.
        """
        assert isinstance(self.pbar, Progress)
        task_id = self.pbar.add_task(name, total=total)
        return task_id

    def update_pbar(self, task_id, advance=1):
        """Updates a certain task in the progress bar.

        Args:
            task_id: ID of the task to update.
            advance: Number of steps advanced onto the target task. (default: 1)
        """
        assert isinstance(self.pbar, Progress)
        if self.pbar.tasks[int(task_id)].finished:
            if self.pbar.tasks[int(task_id)].stop_time is None:
                self.pbar.stop_task(task_id)
        else:
            self.pbar.update(task_id, advance=advance)

    def close_pbar(self):
        """Closes the progress bar"""
        assert isinstance(self.pbar, Progress)
        self.pbar.stop()
        self.pbar = None


class DumbLogger(object):
    """Implements a dumb logger.

    This logger also has member functions like `info()`, `warning()`, etc. But
    nothing will be logged.
    """

    def __init__(self, *_unused_args, **_unused_kwargs):
        """Initializes with dumb functions."""
        self.logger = None
        self.log = lambda *args, **kwargs: None
        self.debug = lambda *args, **kwargs: None
        self.info = lambda *args, **kwargs: None
        self.warning = lambda *args, **kwargs: None
        self.error = lambda *args, **kwargs: None
        self.exception = lambda *args, **kwargs: None
        self.critical = lambda *args, **kwargs: None
        self.print = lambda *args, **kwargs: None

        self.pbar = None
        self.init_pbar = lambda *args, **kwargs: None
        self.add_pbar_task = lambda *args, **kwargs: -1
        self.update_pbar = lambda *args, **kwargs: None
        self.close_pbar = lambda *args, **kwargs: None
