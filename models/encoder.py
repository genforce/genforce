# python 3.7
"""Contains the implementation of encoder network for GAN inversion.

The network structure is primarily based on ResNet. The implementation is
modified from `torchvision.models.resnet`.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['EncoderNet']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Final resolution.
_FINAL_RES = 4

class BasicBlock(nn.Module):
    """Implementation of ResNet BasicBlock."""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 base_width=64,
                 stride=1,
                 groups=1,
                 dilation=1,
                 norm_layer=None,
                 downsample=None):
        super().__init__()
        if base_width != 64:
            raise ValueError(f'BasicBlock of ResNet only supports '
                             f'`base_width=64`, but {base_width} received!')
        if stride not in [1, 2]:
            raise ValueError(f'BasicBlock of ResNet only supports `stride=1` '
                             f'and `stride=2`, but {stride} received!')
        if groups != 1:
            raise ValueError(f'BasicBlock of ResNet only supports `groups=1`, '
                             f'but {groups} received!')
        if dilation != 1:
            raise ValueError(f'BasicBlock of ResNet only supports '
                             f'`dilation=1`, but {dilation} received!')

        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels=inplanes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=1,
                               dilation=1,
                               bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               dilation=1,
                               bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)

        return out


class Bottleneck(nn.Module):
    """Implementation of ResNet Bottleneck."""

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 base_width=64,
                 stride=1,
                 groups=1,
                 dilation=1,
                 norm_layer=None,
                 downsample=None):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f'Bottlenet of ResNet only supports `stride=1` '
                             f'and `stride=2`, but {stride} received!')

        width = int(planes * (base_width / 64)) * groups
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels=inplanes,
                               out_channels=width,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(in_channels=width,
                               out_channels=width,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               groups=groups,
                               dilation=dilation,
                               bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(in_channels=width,
                               out_channels=planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out + identity)

        return out


class FPN(nn.Module):
    """Implementation of Feature Pyramid Network (FPN).

    The input of this module is a pyramid of features with reducing resolutions.
    Then, this module fuses these multi-level features from `top_level` to
    `bottom_level`. In particular, starting from the `top_level`, each feature
    is convoluted, upsampled, and fused into its previous feature (which is also
    convoluted).

    Args:
        pyramid_channels: A list of integers, each of which indicates the number
            of channels of the feature from a particular level.
        out_channels: Number of channels for each output.

    Returns:
        A list of feature maps, each of which has `out_channels` channels.
    """

    def __init__(self, pyramid_channels, out_channels):
        super().__init__()
        assert isinstance(pyramid_channels, (list, tuple))
        self.num_levels = len(pyramid_channels)

        self.lateral_conv_list = nn.ModuleList()
        self.feature_conv_list = nn.ModuleList()
        for i in range(self.num_levels):
            in_channels = pyramid_channels[i]
            self.lateral_conv_list.append(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True))
            self.feature_conv_list.append(nn.Conv2d(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True))

    def forward(self, inputs):
        if len(inputs) != self.num_levels:
            raise ValueError('Number of inputs and `num_levels` mismatch!')

        # Project all related features to `out_channels`.
        laterals = []
        for i in range(self.num_levels):
            laterals.append(self.lateral_conv_list[i](inputs[i]))

        # Fusion, starting from `top_level`.
        for i in range(self.num_levels - 1, 0, -1):
            scale_factor = laterals[i - 1].shape[2] // laterals[i].shape[2]
            laterals[i - 1] = (laterals[i - 1] +
                               F.interpolate(laterals[i],
                                             mode='nearest',
                                             scale_factor=scale_factor))

        # Get outputs.
        outputs = []
        for i, lateral in enumerate(laterals):
            outputs.append(self.feature_conv_list[i](lateral))

        return outputs


class SAM(nn.Module):
    """Implementation of Spatial Alignment Module (SAM).

    The input of this module is a pyramid of features with reducing resolutions.
    Then this module downsamples all levels of feature to the minimum resolution
    and fuses it with the smallest feature map.

    Args:
        pyramid_channels: A list of integers, each of which indicates the number
            of channels of the feature from a particular level.
        out_channels: Number of channels for each output.

    Returns:
        A list of feature maps, each of which has `out_channels` channels.
    """

    def __init__(self, pyramid_channels, out_channels):
        super().__init__()
        assert isinstance(pyramid_channels, (list, tuple))
        self.num_levels = len(pyramid_channels)

        self.fusion_conv_list = nn.ModuleList()
        for i in range(self.num_levels):
            in_channels = pyramid_channels[i]
            self.fusion_conv_list.append(nn.Conv2d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=3,
                                                   padding=1,
                                                   bias=True))

    def forward(self, inputs):
        if len(inputs) != self.num_levels:
            raise ValueError('Number of inputs and `num_levels` mismatch!')

        output_res = inputs[-1].shape[2:]
        for i in range(self.num_levels - 1, -1, -1):
            if i != self.num_levels - 1:
                inputs[i] = F.adaptive_avg_pool2d(inputs[i], output_res)
            inputs[i] = self.fusion_conv_list[i](inputs[i])
            if i != self.num_levels - 1:
                inputs[i] = inputs[i] + inputs[-1]

        return inputs


class CodeHead(nn.Module):
    """Implementation of the task-head to produce inverted codes."""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
        if norm_layer is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_layer(out_channels)

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        latent = self.fc(x)
        latent = latent.unsqueeze(2).unsqueeze(3)
        latent = self.norm(latent)

        return latent.flatten(start_dim=1)

class EncoderNet(nn.Module):
    """Define the ResNet-based encoder network for GAN inversion.

    On top of the backbone, there are several task-heads to produce inverted
    codes. Please use `latent_dim` and `num_latents_per_head` to define the
    structure.

    Settings for the encoder network:

    (1) resolution: The resolution of the output image.
    (2) latent_dim: Dimension of the latent space. A number (one code will be
        produced), or a list of numbers regarding layer-wise latent codes.
    (3) num_latents_per_head: Number of latents that is produced by each head.
    (4) image_channels: Number of channels of the output image. (default: 3)

    ResNet-related settings:

    (1) network_depth: Depth of the network, like 18 for ResNet18. (default: 18)
    (2) inplanes: Number of channels of the first convolutional layer.
        (default: 64)
    (3) groups: Groups of the convolution, used in ResNet. (default: 1)
    (4) width_per_group: Number of channels per group, used in ResNet.
        (default: 64)
    (5) replace_stride_with_dilation: Wether to replace stride with dilation,
        used in ResNet. (default: None)
    (6) norm_layer: Normalization layer used in the encoder.
        (default: nn.BatchNorm2d)
    (7) max_channels: Maximum number of channels in each layer. (default: 512)

    Task-head related settings:

    (1) use_fpn: Whether to use Feature Pyramid Network (FPN) before outputing
        the latent code. (default: True)
    (2) fpn_channels: Number of channels used in FPN. (default: 512)
    (3) use_sam: Whether to use Spatial Alignment Module (SAM) before outputing
        the latent code. (default: True)
    (4) sam_channels: Number of channels used in SAM. (default: 512)
    """

    arch_settings = {
        18: (BasicBlock,  [2, 2, 2, 2]),
        34: (BasicBlock,  [3, 4, 6, 3]),
        50: (Bottleneck,  [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3])
    }

    def __init__(self,
                 resolution,
                 latent_dim,
                 num_latents_per_head,
                 image_channels=3,
                 network_depth=18,
                 inplanes=64,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm2d,
                 max_channels=512,
                 use_fpn=True,
                 fpn_channels=512,
                 use_sam=True,
                 sam_channels=512):
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if network_depth not in self.arch_settings:
            raise ValueError(f'Invalid network depth: `{network_depth}`!\n'
                             f'Options allowed: '
                             f'{list(self.arch_settings.keys())}.')
        if isinstance(latent_dim, int):
            latent_dim = [latent_dim]
        assert isinstance(latent_dim, (list, tuple))
        assert isinstance(num_latents_per_head, (list, tuple))
        assert sum(num_latents_per_head) == len(latent_dim)

        self.resolution = resolution
        self.latent_dim = latent_dim
        self.num_latents_per_head = num_latents_per_head
        self.num_heads = len(self.num_latents_per_head)
        self.image_channels = image_channels
        self.inplanes = inplanes
        self.network_depth = network_depth
        self.groups = groups
        self.dilation = 1
        self.base_width = width_per_group
        self.replace_stride_with_dilation = replace_stride_with_dilation
        if norm_layer is None or norm_layer == nn.BatchNorm2d:
            norm_layer = nn.SyncBatchNorm
        self.norm_layer = norm_layer
        self.max_channels = max_channels
        self.use_fpn = use_fpn
        self.fpn_channels = fpn_channels
        self.use_sam = use_sam
        self.sam_channels = sam_channels

        block_fn, num_blocks_per_stage = self.arch_settings[network_depth]

        self.num_stages = int(np.log2(resolution // _FINAL_RES)) - 1
        for i in range(4, self.num_stages):
            num_blocks_per_stage.append(1)
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False] * self.num_stages

        # Backbone.
        self.conv1 = nn.Conv2d(in_channels=self.image_channels,
                               out_channels=self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage_channels = [self.inplanes]
        for i in range(1, self.num_stages + 1):
            channels = min(self.max_channels, self.inplanes * (2 ** (i - 1)))
            num_blocks = num_blocks_per_stage[i - 1]
            stride = 1 if i == 1 else 2
            dilate = replace_stride_with_dilation[i - 1]
            self.add_module(f'layer{i}',
                            self._make_stage(block_fn=block_fn,
                                             planes=channels,
                                             num_blocks=num_blocks,
                                             stride=stride,
                                             dilate=dilate))
            self.stage_channels.append(channels)

        if self.num_heads > len(self.stage_channels):
            raise ValueError(f'Number of task heads is larger than number of '
                             f'stages! Please reduce the number of heads.')

        # Task-head.
        if self.num_heads == 1:
            self.use_fpn = False
            self.use_sam = False

        if self.use_fpn:
            fpn_pyramid_channels = self.stage_channels[-self.num_heads:]
            self.fpn = FPN(pyramid_channels=fpn_pyramid_channels,
                           out_channels=self.fpn_channels)
        if self.use_sam:
            if use_fpn:
                sam_pyramid_channels = [self.fpn_channels] * self.num_heads
            else:
                sam_pyramid_channels = self.stage_channels[-self.num_heads:]
            self.sam = SAM(pyramid_channels=sam_pyramid_channels,
                           out_channels=self.sam_channels)

        self.head_list = nn.ModuleList()
        for head_idx in range(self.num_heads):
            # Parse in_channels.
            if self.use_sam:
                in_channels = self.sam_channels
            elif self.use_fpn:
                in_channels = self.fpn_channels
            else:
                in_channels = self.stage_channels[head_idx - self.num_heads]
            in_channels = in_channels * _FINAL_RES * _FINAL_RES

            # Parse out_channels.
            start_latent_idx = sum(self.num_latents_per_head[:head_idx])
            end_latent_idx = sum(self.num_latents_per_head[:head_idx + 1])
            out_channels = sum(self.latent_dim[start_latent_idx:end_latent_idx])

            self.head_list.append(CodeHead(in_channels=in_channels,
                                           out_channels=out_channels,
                                           norm_layer=self.norm_layer))

    def _make_stage(self, block_fn, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes,
                          out_channels=planes * block_fn.expansion,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          dilation=1,
                          groups=1,
                          bias=False),
                norm_layer(planes * block_fn.expansion),
            )

        blocks = []
        blocks.append(block_fn(inplanes=self.inplanes,
                               planes=planes,
                               base_width=self.base_width,
                               stride=stride,
                               groups=self.groups,
                               dilation=previous_dilation,
                               norm_layer=norm_layer,
                               downsample=downsample))
        self.inplanes = planes * block_fn.expansion
        for _ in range(1, num_blocks):
            blocks.append(block_fn(inplanes=self.inplanes,
                                   planes=planes,
                                   base_width=self.base_width,
                                   stride=1,
                                   groups=self.groups,
                                   dilation=self.dilation,
                                   norm_layer=norm_layer,
                                   downsample=None))

        return nn.Sequential(*blocks)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = [x]
        for i in range(1, self.num_stages + 1):
            x = getattr(self, f'layer{i}')(x)
            features.append(x)
        features = features[-self.num_heads:]

        if self.use_fpn:
            features = self.fpn(features)
        if self.use_sam:
            features = self.sam(features)
        else:
            final_size = features[-1].shape[2:]
            for i in range(self.num_heads - 1):
                features[i] = F.adaptive_avg_pool2d(features[i], final_size)

        outputs = []
        for head_idx in range(self.num_heads):
            codes = self.head_list[head_idx](features[head_idx])
            start_latent_idx = sum(self.num_latents_per_head[:head_idx])
            end_latent_idx = sum(self.num_latents_per_head[:head_idx + 1])
            split_size = self.latent_dim[start_latent_idx:end_latent_idx]
            outputs.extend(torch.split(codes, split_size, dim=1))
        max_dim = max(self.latent_dim)
        for i, dim in enumerate(self.latent_dim):
            if dim < max_dim:
                outputs[i] = F.pad(outputs[i], (0, max_dim - dim))
            outputs[i] = outputs[i].unsqueeze(1)

        return torch.cat(outputs, dim=1)

    def forward(self, x):
        return self._forward_impl(x)
