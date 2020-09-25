# python3.7
"""Misc utility functions used for model running."""

__all__ = ['format_time']


def format_time(seconds):
    """Formats seconds to readable time string.

    Args:
        seconds: Number of seconds to format.

    Returns:
        The formatted time string.

    Raises:
        ValueError: If the input `seconds` is less than 0.
    """
    if seconds < 0:
        raise ValueError(f'Input `seconds` should be greater than or equal to '
                         f'0, but `{seconds}` is received!')

    # Returns seconds as float if less than 1 minute.
    if seconds < 10:
        return f'{seconds:5.3f}s'
    if seconds < 60:
        return f'{seconds:5.2f}s'

    seconds = int(seconds + 0.5)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f'{days:2d}d{hours:02d}h'
    if hours:
        return f'{hours:2d}h{minutes:02d}m'
    return f'{minutes:2d}m{seconds:02d}s'
