# python3.7
"""Contains the base class for runner.

This runner can be used for both training and inference with multi-threads.
"""

import os
import json
from copy import deepcopy

import torch
import torch.distributed as dist

from datasets import BaseDataset
from datasets import IterDataLoader
from models import build_model
from . import controllers
from . import losses
from . import misc
from .optimizer import build_optimizers
from .running_stats import RunningStats


def _strip_state_dict_prefix(state_dict, prefix='module.'):
    """Removes the name prefix in checkpoint.

    Basically, when the model is deployed in parallel, the prefix `module.` will
    be added to the saved checkpoint. This function is used to remove the
    prefix, which is friendly to checkpoint loading.

    Args:
        state_dict: The state dict where the variable names are processed.
        prefix: The prefix to remove. (default: `module.`)
    """
    if not all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict

    stripped_state_dict = dict()
    for key in state_dict:
        stripped_state_dict[key.replace(prefix, '')] = state_dict[key]
    return stripped_state_dict


class BaseRunner(object):
    """Defines the base runner class."""

    def __init__(self, config, logger):
        self._name = self.__class__.__name__
        self._config = deepcopy(config)
        self.logger = logger
        self.work_dir = self.config.work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        self.logger.info('Running Configuration:')
        config_str = json.dumps(self.config, indent=4).replace('"', '\'')
        self.logger.print(config_str + '\n')
        with open(os.path.join(self.work_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        self.batch_size = self.config.batch_size
        self.val_batch_size = self.config.get('val_batch_size', self.batch_size)
        self._iter = 0
        self._start_iter = 0
        self.seen_img = 0
        self.total_iters = self.config.get('total_iters', 0)
        if self.total_iters == 0 and self.config.get('total_img', 0) > 0:
            total_image = self.config.get('total_img')
            total_batch = self.world_size * self.batch_size
            self.total_iters = int(total_image / total_batch + 0.5)

        self.mode = None
        self.train_loader = None
        self.val_loader = None

        self.models = dict()
        self.optimizers = dict()
        self.lr_schedulers = dict()
        self.controllers = []
        self.loss = None

        self.running_stats = RunningStats()
        self.start_time = 0
        self.end_time = 0
        self.timer = controllers.Timer()
        self.timer.start(self)

        self.build_models()
        self.build_controllers()

    def finish(self):
        """Finishes runner by ending controllers and timer."""
        for controller in self.controllers:
            controller.end(self)
        self.timer.end(self)
        self.logger.info(f'Finish runner in '
                         f'{misc.format_time(self.end_time - self.start_time)}')

    @property
    def name(self):
        """Returns the name of the runner."""
        return self._name

    @property
    def config(self):
        """Returns the configuration of the runner."""
        return self._config

    @property
    def rank(self):
        """Returns the rank of the current runner."""
        return self._rank

    @property
    def world_size(self):
        """Returns the world size."""
        return self._world_size

    @property
    def iter(self):
        """Returns the current iteration."""
        return self._iter

    @property
    def start_iter(self):
        """Returns the start iteration."""
        return self._start_iter

    def convert_epoch_to_iter(self, epoch):
        """Converts number of epochs to number of iterations."""
        return int(epoch * len(self.train_loader) + 0.5)

    def build_dataset(self, mode):
        """Builds train/val dataset."""
        if not hasattr(self.config, 'data'):
            return
        assert isinstance(mode, str)
        mode = mode.lower()
        self.logger.info(f'Building `{mode}` dataset ...')
        if mode not in ['train', 'val']:
            raise ValueError(f'Invalid dataset mode `{mode}`!')
        dataset = BaseDataset(**self.config.data[mode])
        if mode == 'train':
            self.train_loader = IterDataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config.data.get('num_workers', 2),
                current_iter=self.iter,
                repeat=self.config.data.get('repeat', 1))
        elif mode == 'val':
            self.val_loader = IterDataLoader(
                dataset=dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.config.data.get('num_workers', 2),
                current_iter=0,
                repeat=1)
        else:
            raise NotImplementedError(f'Not implemented dataset mode `{mode}`!')
        self.logger.info(f'Finish building `{mode}` dataset.')

    def build_models(self):
        """Builds models, optimizers, and learning rate schedulers."""
        self.logger.info(f'Building models ...')
        lr_config = dict()
        opt_config = dict()
        for module, module_config in self.config.modules.items():
            model_config = module_config['model']
            self.models[module] = build_model(module=module, **model_config)
            self.models[module].cuda()
            opt_config[module] = module_config.get('opt', None)
            lr_config[module] = module_config.get('lr', None)
        build_optimizers(opt_config, self)
        self.controllers.append(controllers.LRScheduler(lr_config))
        self.logger.info(f'Finish building models.')

        model_info = 'Model structures:\n'
        model_info += '==============================================\n'
        for module in self.models:
            model_info += f'{module}\n'
            model_info += '----------------------------------------------\n'
            model_info += str(self.models[module])
            model_info += '\n'
            model_info += "==============================================\n"
        self.logger.info(model_info)

    def distribute(self):
        """Sets `self.model` as `torch.nn.parallel.DistributedDataParallel`."""
        for name in self.models:
            self.models[name] = torch.nn.parallel.DistributedDataParallel(
                module=self.models[name],
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=True)

    @staticmethod
    def get_module(model):
        """Handles distributed model."""
        if hasattr(model, 'module'):
            return model.module
        return model

    def build_controllers(self):
        """Builds additional controllers besides LRScheduler."""
        if not hasattr(self.config, 'controllers'):
            return
        self.logger.info(f'Building controllers ...')
        for key, ctrl_config in self.config.controllers.items():
            self.controllers.append(getattr(controllers, key)(ctrl_config))
        self.controllers.sort(key=lambda x: x.priority)
        for controller in self.controllers:
            controller.start(self)
        self.logger.info(f'Finish building controllers.')

    def build_loss(self):
        """Builds loss functions."""
        if not hasattr(self.config, 'loss'):
            return
        self.logger.info(f'Building loss function ...')
        loss_config = deepcopy(self.config.loss)
        loss_type = loss_config.pop('type')
        self.loss = getattr(losses, loss_type)(self, **loss_config)
        self.logger.info(f'Finish building loss function.')

    def pre_execute_controllers(self):
        """Pre-executes all controllers in order of priority."""
        for controller in self.controllers:
            controller.pre_execute(self)

    def post_execute_controllers(self):
        """Post-executes all controllers in order of priority."""
        for controller in self.controllers:
            controller.post_execute(self)

    def cpu(self):
        """Puts models to CPU."""
        for name in self.models:
            self.models[name].cpu()

    def cuda(self):
        """Puts models to CUDA."""
        for name in self.models:
            self.models[name].cuda()

    def set_model_requires_grad(self, name, requires_grad):
        """Sets the `requires_grad` configuration for a particular model."""
        for param in self.models[name].parameters():
            param.requires_grad = requires_grad

    def set_models_requires_grad(self, requires_grad):
        """Sets the `requires_grad` configuration for all models."""
        for name in self.models:
            self.set_model_requires_grad(name, requires_grad)

    def set_model_mode(self, name, mode):
        """Sets the `train/val` mode for a particular model."""
        if isinstance(mode, str):
            mode = mode.lower()
        if mode == 'train' or mode is True:
            self.models[name].train()
        elif mode in ['val', 'test', 'eval'] or mode is False:
            self.models[name].eval()
        else:
            raise ValueError(f'Invalid model mode `{mode}`!')

    def set_mode(self, mode):
        """Sets the `train/val` mode for all models."""
        self.mode = mode
        for name in self.models:
            self.set_model_mode(name, mode)

    def train_step(self, data, **train_kwargs):
        """Executes one training step."""
        raise NotImplementedError('Should be implemented in derived class.')

    def train(self, **train_kwargs):
        """Training function."""
        self.set_mode('train')
        self.distribute()
        self.build_dataset('train')
        self.build_loss()

        self.logger.print()
        self.logger.info(f'Start training.')
        if self.total_iters == 0:
            total_epochs = self.config.get('total_epochs', 0)
            self.total_iters = self.convert_epoch_to_iter(total_epochs)
        assert self.total_iters > 0
        while self.iter < self.total_iters:
            self._iter += 1
            self.pre_execute_controllers()
            data_batch = next(self.train_loader)
            self.timer.pre_execute(self)
            for key in data_batch:
                assert data_batch[key].shape[0] == self.batch_size
                data_batch[key] = data_batch[key].cuda(
                    torch.cuda.current_device(), non_blocking=True)
            self.train_step(data_batch, **train_kwargs)
            self.seen_img += self.batch_size * self.world_size
            self.timer.post_execute(self)
            self.post_execute_controllers()
        self.finish()

    def val(self, **val_kwargs):
        """Validation function."""
        raise NotImplementedError('Should be implemented in derived class.')

    def save(self,
             filepath,
             running_metadata=True,
             learning_rate=True,
             optimizer=True,
             running_stats=False):
        """Saves the current running status.
        Args:
            filepath: File path to save the checkpoint.
            running_metadata: Whether to save the running metadata, such as
                batch size, current iteration, etc. (default: True)
            learning_rate: Whether to save the learning rate. (default: True)
            optimizer: Whether to save the optimizer. (default: True)
            running_stats: Whether to save the running stats. (default: False)
        """
        checkpoint = dict()
        # Models.
        checkpoint['models'] = dict()
        for name, model in self.models.items():
            checkpoint['models'][name] = self.get_module(model).state_dict()
        # Running metadata.
        if running_metadata:
            checkpoint['running_metadata'] = {
                'iter': self.iter,
                'seen_img': self.seen_img,
            }
        # Optimizers.
        if optimizer:
            checkpoint['optimizers'] = dict()
            for opt_name, opt in self.optimizers.items():
                checkpoint['optimizers'][opt_name] = opt.state_dict()
        # Learning rates.
        if learning_rate:
            checkpoint['learning_rates'] = dict()
            for lr_name, lr in self.lr_schedulers.items():
                checkpoint['learning_rates'][lr_name] = lr.state_dict()
        # Running stats.
        # TODO: Test saving and loading running stats.
        if running_stats:
            checkpoint['running_stats'] = self.running_stats
        # Save checkpoint.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        self.logger.info(f'Successfully saved checkpoint to `{filepath}`.')

    def load(self,
             filepath,
             running_metadata=True,
             learning_rate=True,
             optimizer=True,
             running_stats=False,
             map_location='cpu'):
        """Loads previous running status.

        Args:
            filepath: File path to load the checkpoint.
            running_metadata: Whether to load the running metadata, such as
                batch size, current iteration, etc. (default: True)
            learning_rate: Whether to load the learning rate. (default: True)
            optimizer: Whether to load the optimizer. (default: True)
            running_stats: Whether to load the running stats. (default: False)
            map_location: Map location used for model loading. (default: `cpu`)
        """
        self.logger.info(f'Resuming from checkpoint `{filepath}` ...')
        if not os.path.isfile(filepath):
            raise IOError(f'Checkpoint `{filepath}` does not exist!')
        map_location = map_location.lower()
        assert map_location in ['cpu', 'gpu']
        if map_location == 'gpu':
            device = torch.cuda.current_device()
            map_location = lambda storage, location: storage.cuda(device)
        checkpoint = torch.load(filepath, map_location=map_location)
        # Load models.
        if 'models' not in checkpoint:
            checkpoint = {'models': checkpoint}
        for model_name, model in self.models.items():
            if model_name not in checkpoint['models']:
                self.logger.warning(f'Model `{model_name}` is not included in '
                                    f'the checkpoint, and hence will NOT be '
                                    f'loaded!')
                continue
            state_dict = _strip_state_dict_prefix(
                checkpoint['models'][model_name])
            model.load_state_dict(state_dict)
            self.logger.info(f'  Successfully loaded model `{model_name}`.')
        # Load running metedata.
        if running_metadata:
            if 'running_metadata' not in checkpoint:
                self.logger.warning(f'Running metadata is not included in the '
                                    f'checkpoint, and hence will NOT be '
                                    f'loaded!')
            else:
                self._iter = checkpoint['running_metadata']['iter']
                self._start_iter = self._iter
                self.seen_img = checkpoint['running_metadata']['seen_img']
        # Load optimizers.
        if optimizer:
            if 'optimizers' not in checkpoint:
                self.logger.warning(f'Optimizers are not included in the '
                                    f'checkpoint, and hence will NOT be '
                                    f'loaded!')
            else:
                for opt_name, opt in self.optimizers.items():
                    if opt_name not in checkpoint['optimizers']:
                        self.logger.warning(f'Optimizer `{opt_name}` is not '
                                            f'included in the checkpoint, and '
                                            f'hence will NOT be loaded!')
                        continue
                    opt.load_state_dict(checkpoint['optimizers'][opt_name])
                    self.logger.info(f'  Successfully loaded optimizer '
                                     f'`{opt_name}`.')
        # Load learning rates.
        if learning_rate:
            if 'learning_rates' not in checkpoint:
                self.logger.warning(f'Learning rates are not included in the '
                                    f'checkpoint, and hence will NOT be '
                                    f'loaded!')
            else:
                for lr_name, lr in self.lr_schedulers.items():
                    if lr_name not in checkpoint['learning_rates']:
                        self.logger.warning(f'Learning rate `{lr_name}` is not '
                                            f'included in the checkpoint, and '
                                            f'hence will NOT be loaded!')
                        continue
                    lr.load_state_dict(checkpoint['learning_rates'][lr_name])
                    self.logger.info(f'  Successfully loaded learning rate '
                                     f'`{lr_name}`.')
        # Load running stats.
        if running_stats:
            if 'running_stats' not in checkpoint:
                self.logger.warning(f'Running stats is not included in the '
                                    f'checkpoint, and hence will NOT be '
                                    f'loaded!')
            else:
                self.running_stats = deepcopy(checkpoint['running_stats'])
                self.logger.info(f'  Successfully loaded running stats.')
        # Log message.
        tailing_message = ''
        if running_metadata and 'running_metadata' in checkpoint:
            tailing_message = f' (iteration {self.iter})'
        self.logger.info(f'Successfully resumed from checkpoint `{filepath}`.'
                         f'{tailing_message}')
