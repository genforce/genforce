# python3.7
"""Unit test for logger."""

import time

from .logger import build_logger


def test_logger():
    """Test function."""
    logger = build_logger('normal', logger_name='normal', logfile_name='')
    rich_logger = build_logger('rich', logger_name='rich', logfile_name='')
    dumb_logger = build_logger('dumb', logger_name='dumb', logfile_name='')

    print('-------------------------------')
    print('| Test `utils.logger.Logger`. |')
    print('-------------------------------')
    logger.print('log')
    logger.debug('log')
    logger.info('log')
    logger.warning('log')
    logger.init_pbar()
    task1 = logger.add_pbar_task('Task 1', 500)
    task2 = logger.add_pbar_task('Task 2', 1000)
    for _ in range(1000):
        logger.update_pbar(task1, 1)
        logger.update_pbar(task2, 1)
        time.sleep(0.005)
    logger.close_pbar()
    print('Success!')

    print('-----------------------------------')
    print('| Test `utils.logger.RichLogger`. |')
    print('-----------------------------------')
    rich_logger.print('rich_log')
    rich_logger.debug('rich_log')
    rich_logger.info('rich_log')
    rich_logger.warning('rich_log')
    rich_logger.init_pbar()
    task1 = rich_logger.add_pbar_task('Rich Task 1', 500)
    task2 = rich_logger.add_pbar_task('Rich Task 2', 1000)
    for _ in range(1000):
        rich_logger.update_pbar(task1, 1)
        rich_logger.update_pbar(task2, 1)
        time.sleep(0.005)
    rich_logger.close_pbar()
    print('Success!')

    print('-----------------------------------')
    print('| Test `utils.logger.DumbLogger`. |')
    print('-----------------------------------')
    dumb_logger.print('dumb_log')
    dumb_logger.debug('dumb_log')
    dumb_logger.info('dumb_log')
    dumb_logger.warning('dumb_log')
    dumb_logger.init_pbar()
    task1 = dumb_logger.add_pbar_task('Dumb Task 1', 500)
    task2 = dumb_logger.add_pbar_task('Dumb Task 2', 1000)
    for _ in range(1000):
        dumb_logger.update_pbar(task1, 1)
        dumb_logger.update_pbar(task2, 1)
        time.sleep(0.005)
    dumb_logger.close_pbar()
    print('Success!')
