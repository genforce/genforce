# python3.7
"""Main function for model training."""

import os.path
import shutil
import warnings
import random
import argparse
import numpy as np

import torch
import torch.distributed as dist

import runners
from utils.logger import build_logger
from utils.misc import init_dist
from utils.misc import DictAction, parse_config, update_config


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Run model training.')
    parser.add_argument('config', type=str,
                        help='Path to the training configuration.')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='The work directory to save logs and checkpoints.')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to the checkpoint to resume training.')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to the checkpoint to load model weights, '
                             'but not resume other states.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed. (default: %(default)s)')
    parser.add_argument('--launcher', type=str, default='pytorch',
                        choices=['pytorch', 'slurm'],
                        help='Launcher type. (default: %(default)s)')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='Backend for distributed launcher. (default: '
                             '%(default)s)')
    parser.add_argument('--rank', type=int, default=-1,
                        help='Node rank for distributed running. (default: '
                             '%(default)s)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Rank of the current node. (default: %(default)s)')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='arguments in dict')
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments.
    args = parse_args()

    # Parse configurations.
    config = parse_config(args.config)
    config = update_config(config, args.options)
    config.work_dir = args.work_dir
    config.resume_path = args.resume_path
    config.weight_path = args.weight_path
    config.seed = args.seed
    config.launcher = args.launcher
    config.backend = args.backend

    # Set CUDNN.
    config.cudnn_benchmark = config.get('cudnn_benchmark', True)
    config.cudnn_deterministic = config.get('cudnn_deterministic', False)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic

    # Set deterministic if random seed is provided.
    if config.seed is not None:
        config.cudnn_deterministic = True
        torch.backends.cudnn.deterministic = True
        warnings.warn('Random seed is set for training! '
                      'This will turn on the CUDNN deterministic setting, '
                      'which may slow down the training considerably! '
                      'Unexpected behavior can be observed when resuming from '
                      'checkpoints.')

    # Set launcher.
    config.is_distributed = True
    init_dist(config.launcher, backend=config.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    config.num_gpus = world_size

    # Set random seed.
    if config.seed is not None:
        random.seed(config.seed * world_size + rank)
        np.random.seed(config.seed * world_size + rank)
        torch.manual_seed(config.seed * world_size + rank)

    # Setup logger.
    if dist.get_rank() == 0:
        logger_type = config.get('logger_type', 'normal')
        logger = build_logger(logger_type, work_dir=config.work_dir)
        shutil.copy(args.config, os.path.join(config.work_dir, 'config.py'))
        commit_id = os.popen('git rev-parse HEAD').readline()
        logger.info(f'Commit ID: {commit_id}')
    else:
        logger = build_logger('dumb', work_dir=config.work_dir)

    # Start training.
    runner = getattr(runners, config.runner_type)(config, logger)
    if config.resume_path:
        runner.load(filepath=config.resume_path,
                    running_metadata=True,
                    learning_rate=True,
                    optimizer=True,
                    running_stats=False)
    if config.weight_path:
        runner.load(filepath=config.weight_path,
                    running_metadata=False,
                    learning_rate=False,
                    optimizer=False,
                    running_stats=False)
    runner.train()


if __name__ == '__main__':
    main()
