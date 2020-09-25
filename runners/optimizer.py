# python3.7
"""Contains the function to build optimizer for runner."""

import math

import torch

__all__ = ['build_optimizer', 'build_optimizers']

_ALLOWED_OPT_TYPES = ['SGD', 'ADAM']


def build_optimizer(config, model):
    """Builds an optimizer for the given model.

    Basically, the configuration is expected to contain following settings:

    (1) opt_type: The type of the optimizer. (required)
    (2) base_lr: The base learning rate for all parameters. (required)
    (3) base_wd: The base weight decay for all parameters. (default: 0.0)
    (4) bias_lr_multiplier: The learning rate multiplier for bias parameters.
        (default: 1.0)
    (5) bias_wd_multiplier: The weight decay multiplier for bias parameters.
        (default: 1.0)
    (6) **kwargs: Additional settings for the optimizer, such as `momentum`.

    Args:
        config: The configuration used to build the optimizer.
        model: The model which the optimizer serves.

    Returns:
        A `torch.optim.Optimizer`.

    Raises:
        ValueError: The `opt_type` is not supported.
        NotImplementedError: If `opt_type` is not implemented.
    """
    assert isinstance(config, dict)
    opt_type = config['opt_type'].upper()
    base_lr = config['base_lr']
    base_wd = config.get('base_wd', 0.0)
    bias_lr_multiplier = config.get('bias_lr_multiplier', 1.0)
    bias_wd_multiplier = config.get('bias_wd_multiplier', 1.0)

    if opt_type not in _ALLOWED_OPT_TYPES:
        raise ValueError(f'Invalid optimizer type `{opt_type}`!'
                         f'Allowed types: {_ALLOWED_OPT_TYPES}.')

    model_params = []
    for param_name, param in model.named_parameters():
        param_group = {'params': [param]}
        if param.requires_grad:
            if 'bias' in param_name:
                param_group['lr'] = base_lr * bias_lr_multiplier
                param_group['weight_decay'] = base_wd * bias_wd_multiplier
            else:
                param_group['lr'] = base_lr
                param_group['weight_decay'] = base_wd
        model_params.append(param_group)

    if opt_type == 'SGD':
        return torch.optim.SGD(params=model_params,
                               lr=base_lr,
                               momentum=config.get('momentum', 0.9),
                               dampening=config.get('dampening', 0),
                               weight_decay=base_wd,
                               nesterov=config.get('nesterov', False))
    if opt_type == 'ADAM':
        return AdamOptimizer(params=model_params,
                             lr=base_lr,
                             betas=config.get('betas', (0.9, 0.999)),
                             eps=config.get('eps', 1e-8),
                             weight_decay=base_wd,
                             amsgrad=config.get('amsgrad', False))
    raise NotImplementedError(f'Not implemented optimizer type `{opt_type}`!')


def build_optimizers(opt_config, runner):
    """Builds optimizers for the given runner.

    The `opt_config` should be a dictionary, where keys are model names and
    each value is the optimizer configuration for a particumar model. All built
    optimizers will be saved in `runner.optimizers`, which is also a dictionary.

    NOTE: The model names should match the keys of `runner.models`.

    Args:
        opt_config: The configuration to build the optimizers.
        runner: The runner to build the optimizer for.
    """
    if not opt_config:
        return

    assert isinstance(opt_config, dict)
    for name, config in opt_config.items():
        if not name or not config:
            continue
        if name in runner.optimizers:
            raise AttributeError(f'Optimizer `{name}` has already existed!')
        if name not in runner.models:
            raise AttributeError(f'Model `{name}` is missing!')
        runner.optimizers[name] = build_optimizer(config, runner.models[name])


# We slightly modify the Adam optimizer from `torch.optim`. since there exists
# some discrepancies between the `torch.optim` version and the TensorFlow
# version. The main difference is where to add the `epsilon`.
# TODO: The modified optimizer does not support `amsgrad` any more.

# pylint: disable=line-too-long
# pylint: disable=unneeded-not
# pylint: disable=misplaced-comparison-constant
# pylint: disable=super-with-arguments

class AdamOptimizer(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                assert not amsgrad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # if amsgrad:
                #     max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # else:
                #     denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # step_size = group['lr'] / bias_correction1

                # p.addcdiv_(exp_avg, denom, value=-step_size)

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']) , value=-step_size)

        return loss

# pylint: enable=line-too-long
# pylint: enable=unneeded-not
# pylint: enable=misplaced-comparison-constant
# pylint: enable=super-with-arguments
