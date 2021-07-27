# python3.7
"""Unit test for logger."""

import time

from .logger import build_logger


def test_logger():
    """Test function."""

    for logger_type in ['normal', 'rich', 'dumb']:
        if logger_type == 'normal':
            class_name = 'Logger'
        elif logger_type == 'rich':
            class_name = 'RichLogger'
        elif logger_type == 'dumb':
            class_name = 'DumbLogger'

        print(f'===== Test `utils.logger.{class_name}` =====')
        logger = build_logger(logger_type,
                              logger_name=logger_type,
                              logfile_name=f'test_{logger_type}_logger.log')
        logger.print('print log')
        logger.debug('debug log')
        logger.info('info log')
        logger.warning('warning log')
        logger.init_pbar()
        task1 = logger.add_pbar_task('Task 1', 500)
        task2 = logger.add_pbar_task('Task 2', 1000)
        for _ in range(1000):
            logger.update_pbar(task1, 1)
            logger.update_pbar(task2, 1)
            time.sleep(0.005)
        logger.close_pbar()
        print('Success!')
