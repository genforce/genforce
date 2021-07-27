# python 3.7
"""Contains the VGG16 model for perceptual feature extraction.

This file is particularly used for computing perceptual loss and hence is highly
recommended to use with pre-trained weights.

The PyTorch weights can be downloaded from

https://drive.google.com/file/d/1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y/view?usp=sharing

which is converted from the Keras model

https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

The variable mapping is shown below

pth_to_tf_var_mapping = {
    'layer0.weight':  'block1_conv1_W_1:0',  # [64, 3, 3, 3]
    'layer0.bias':    'block1_conv1_b_1:0',  # [64]
    'layer2.weight':  'block1_conv2_W_1:0',  # [64, 64, 3, 3]
    'layer2.bias':    'block1_conv2_b_1:0',  # [64]
    'layer5.weight':  'block2_conv1_W_1:0',  # [128, 64, 3, 3]
    'layer5.bias':    'block2_conv1_b_1:0',  # [128]
    'layer7.weight':  'block2_conv2_W_1:0',  # [128, 128, 3, 3]
    'layer7.bias':    'block2_conv2_b_1:0',  # [128]
    'layer10.weight': 'block3_conv1_W_1:0',  # [256, 128, 3, 3]
    'layer10.bias':   'block3_conv1_b_1:0',  # [256]
    'layer12.weight': 'block3_conv2_W_1:0',  # [256, 256, 3, 3]
    'layer12.bias':   'block3_conv2_b_1:0',  # [256]
    'layer14.weight': 'block3_conv3_W_1:0',  # [256, 256, 3, 3]
    'layer14.bias':   'block3_conv3_b_1:0',  # [256]
    'layer17.weight': 'block4_conv1_W_1:0',  # [512, 256, 3, 3]
    'layer17.bias':   'block4_conv1_b_1:0',  # [512]
    'layer19.weight': 'block4_conv2_W_1:0',  # [512, 512, 3, 3]
    'layer19.bias':   'block4_conv2_b_1:0',  # [512]
    'layer21.weight': 'block4_conv3_W_1:0',  # [512, 512, 3, 3]
    'layer21.bias':   'block4_conv3_b_1:0',  # [512]
    'layer24.weight': 'block5_conv1_W_1:0',  # [512, 512, 3, 3]
    'layer24.bias':   'block5_conv1_b_1:0',  # [512]
    'layer26.weight': 'block5_conv2_W_1:0',  # [512, 512, 3, 3]
    'layer26.bias':   'block5_conv2_b_1:0',  # [512]
    'layer28.weight': 'block5_conv3_W_1:0',  # [512, 512, 3, 3]
    'layer28.bias':   'block5_conv3_b_1:0',  # [512]
}
"""

import os
import warnings
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

__all__ = ['PerceptualModel']

_MEAN_STATS = (103.939, 116.779, 123.68)


class PerceptualModel(nn.Module):
    """Defines the VGG16 structure as the perceptual network.

    This model takes `RGB` images with data format `NCHW` as the raw inputs, and
    outputs the perceptual feature. This following operations will be performed
    to preprocess the inputs to match the preprocessing during the model
    training:
    (1) Shift pixel range to [0, 255].
    (2) Change channel order to `BGR`.
    (3) Subtract the statistical mean.

    NOTE: The three fully connected layers on top of the model are dropped.
    """

    def __init__(self,
                 output_layer_idx=23,
                 min_val=-1.0,
                 max_val=1.0,
                 pretrained_weight_path=None):
        """Defines the network structure.

        Args:
            output_layer_idx: Index of layer whose output will be used as the
                perceptual feature. (default: 23, which is the `block4_conv3`
                layer activated by `ReLU` function)
            min_val: Minimum value of the raw input. (default: -1.0)
            max_val: Maximum value of the raw input. (default: 1.0)
            pretrained_weight_path: Path to the pretrained weights.
                (default: None)
        """
        super().__init__()
        self.vgg16 = nn.Sequential(OrderedDict({
            'layer0': nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            'layer1': nn.ReLU(inplace=True),
            'layer2': nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            'layer3': nn.ReLU(inplace=True),
            'layer4': nn.MaxPool2d(kernel_size=2, stride=2),
            'layer5': nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            'layer6': nn.ReLU(inplace=True),
            'layer7': nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            'layer8': nn.ReLU(inplace=True),
            'layer9': nn.MaxPool2d(kernel_size=2, stride=2),
            'layer10': nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            'layer11': nn.ReLU(inplace=True),
            'layer12': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            'layer13': nn.ReLU(inplace=True),
            'layer14': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            'layer15': nn.ReLU(inplace=True),
            'layer16': nn.MaxPool2d(kernel_size=2, stride=2),
            'layer17': nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            'layer18': nn.ReLU(inplace=True),
            'layer19': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer20': nn.ReLU(inplace=True),
            'layer21': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer22': nn.ReLU(inplace=True),
            'layer23': nn.MaxPool2d(kernel_size=2, stride=2),
            'layer24': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer25': nn.ReLU(inplace=True),
            'layer26': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer27': nn.ReLU(inplace=True),
            'layer28': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer29': nn.ReLU(inplace=True),
            'layer30': nn.MaxPool2d(kernel_size=2, stride=2),
        }))
        self.output_layer_idx = output_layer_idx
        self.min_val = min_val
        self.max_val = max_val
        self.mean = torch.from_numpy(np.array(_MEAN_STATS)).view(1, 3, 1, 1)
        self.mean = self.mean.type(torch.FloatTensor)

        self.pretrained_weight_path = pretrained_weight_path
        if os.path.isfile(self.pretrained_weight_path):
            self.vgg16.load_state_dict(
                torch.load(self.pretrained_weight_path, map_location='cpu'))
        else:
            warnings.warn('No pre-trained weights found for perceptual model!')

    def forward(self, x):
        x = (x - self.min_val) * 255.0 / (self.max_val - self.min_val)
        x = x.flip(1)  # RGB to BGR
        x = x - self.mean.to(x)
        # TODO: Resize image?
        for idx, layer in enumerate(self.vgg16.children()):
            if idx == self.output_layer_idx:
                break
            x = layer(x)
        # x = x.permute(0, 2, 3, 1)
        x = x.flatten(start_dim=1)
        return x
