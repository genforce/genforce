# python3.7
"""Converts StyleGAN2-ADA-PyTorch model to match this repository.

The models can be trained through OR released by the repository:

https://github.com/NVlabs/stylegan2-ada-pytorch
"""

import os
import sys
import re
import pickle
import warnings
from tqdm import tqdm
import numpy as np

import torch

from models import build_model
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import postprocess_image

__all__ = ['convert_stylegan2ada_pth_weight']

GAN_TPYE = 'stylegan2'
OFFICIAL_CODE_DIR = 'stylegan2ada_pth_official'
BASE_DIR = os.path.dirname(os.path.relpath(__file__))
CODE_PATH = os.path.join(BASE_DIR, OFFICIAL_CODE_DIR)

TRUNC_PSI = 0.5
TRUNC_LAYERS = 18
RANDOMIZE_NOISE = False
NOISE_MODE = 'random' if RANDOMIZE_NOISE else 'const'

# The following two dictionary of mapping patterns are modified from
# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/legacy.py
G_PTH_TO_TF_VAR_MAPPING_PATTERN = {
    r'mapping\.w_avg':
        lambda:   f'dlatent_avg',
    r'mapping\.embed\.weight':
        lambda:   f'LabelEmbed/weight',
    r'mapping\.embed\.bias':
        lambda:   f'LabelEmbed/bias',
    r'mapping\.fc(\d+)\.weight':
        lambda i: f'Dense{i}/weight',
    r'mapping\.fc(\d+)\.bias':
        lambda i: f'Dense{i}/bias',
    r'synthesis\.b4\.const':
        lambda:   f'4x4/Const/const',
    r'synthesis\.b4\.conv1\.weight':
        lambda:   f'4x4/Conv/weight',
    r'synthesis\.b4\.conv1\.bias':
        lambda:   f'4x4/Conv/bias',
    r'synthesis\.b4\.conv1\.noise_const':
        lambda:   f'noise0',
    r'synthesis\.b4\.conv1\.noise_strength':
        lambda:   f'4x4/Conv/noise_strength',
    r'synthesis\.b4\.conv1\.affine\.weight':
        lambda:   f'4x4/Conv/mod_weight',
    r'synthesis\.b4\.conv1\.affine\.bias':
        lambda:   f'4x4/Conv/mod_bias',
    r'synthesis\.b(\d+)\.conv0\.weight':
        lambda r: f'{r}x{r}/Conv0_up/weight',
    r'synthesis\.b(\d+)\.conv0\.bias':
        lambda r: f'{r}x{r}/Conv0_up/bias',
    r'synthesis\.b(\d+)\.conv0\.noise_const':
        lambda r: f'noise{int(np.log2(int(r)))*2-5}',
    r'synthesis\.b(\d+)\.conv0\.noise_strength':
        lambda r: f'{r}x{r}/Conv0_up/noise_strength',
    r'synthesis\.b(\d+)\.conv0\.affine\.weight':
        lambda r: f'{r}x{r}/Conv0_up/mod_weight',
    r'synthesis\.b(\d+)\.conv0\.affine\.bias':
        lambda r: f'{r}x{r}/Conv0_up/mod_bias',
    r'synthesis\.b(\d+)\.conv1\.weight':
        lambda r: f'{r}x{r}/Conv1/weight',
    r'synthesis\.b(\d+)\.conv1\.bias':
        lambda r: f'{r}x{r}/Conv1/bias',
    r'synthesis\.b(\d+)\.conv1\.noise_const':
        lambda r: f'noise{int(np.log2(int(r)))*2-4}',
    r'synthesis\.b(\d+)\.conv1\.noise_strength':
        lambda r: f'{r}x{r}/Conv1/noise_strength',
    r'synthesis\.b(\d+)\.conv1\.affine\.weight':
        lambda r: f'{r}x{r}/Conv1/mod_weight',
    r'synthesis\.b(\d+)\.conv1\.affine\.bias':
        lambda r: f'{r}x{r}/Conv1/mod_bias',
    r'synthesis\.b(\d+)\.torgb\.weight':
        lambda r: f'{r}x{r}/ToRGB/weight',
    r'synthesis\.b(\d+)\.torgb\.bias':
        lambda r: f'{r}x{r}/ToRGB/bias',
    r'synthesis\.b(\d+)\.torgb\.affine\.weight':
        lambda r: f'{r}x{r}/ToRGB/mod_weight',
    r'synthesis\.b(\d+)\.torgb\.affine\.bias':
        lambda r: f'{r}x{r}/ToRGB/mod_bias',
    r'synthesis\.b(\d+)\.skip\.weight':
        lambda r: f'{r}x{r}/Skip/weight',
    r'.*\.resample_filter':
        None,
}
D_PTH_TO_TF_VAR_MAPPING_PATTERN = {
    r'b(\d+)\.fromrgb\.weight':
        lambda r:    f'{r}x{r}/FromRGB/weight',
    r'b(\d+)\.fromrgb\.bias':
        lambda r:    f'{r}x{r}/FromRGB/bias',
    r'b(\d+)\.conv(\d+)\.weight':
        lambda r, i: f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/weight',
    r'b(\d+)\.conv(\d+)\.bias':
        lambda r, i: f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/bias',
    r'b(\d+)\.skip\.weight':
        lambda r:    f'{r}x{r}/Skip/weight',
    r'mapping\.embed\.weight':
        lambda:      f'LabelEmbed/weight',
    r'mapping\.embed\.bias':
        lambda:      f'LabelEmbed/bias',
    r'mapping\.fc(\d+)\.weight':
        lambda i:    f'Mapping{i}/weight',
    r'mapping\.fc(\d+)\.bias':
        lambda i:    f'Mapping{i}/bias',
    r'b4\.conv\.weight':
        lambda:      f'4x4/Conv/weight',
    r'b4\.conv\.bias':
        lambda:      f'4x4/Conv/bias',
    r'b4\.fc\.weight':
        lambda:      f'4x4/Dense0/weight',
    r'b4\.fc\.bias':
        lambda:      f'4x4/Dense0/bias',
    r'b4\.out\.weight':
        lambda:      f'Output/weight',
    r'b4\.out\.bias':
        lambda:      f'Output/bias',
    r'.*\.resample_filter':
        None,
}


def convert_stylegan2ada_pth_weight(src_weight_path,
                                    dst_weight_path,
                                    test_num=10,
                                    save_test_image=False,
                                    verbose=False):
    """Converts the pre-trained StyleGAN2-ADA-PyTorch weights.

    Args:
        src_weight_path: Path to the source model to load weights from.
        dst_weight_path: Path to the target model to save converted weights.
        test_num: Number of samples used to test the conversion. (default: 10)
        save_test_image: Whether to save the test images. (default: False)
        verbose: Whether to print verbose log message. (default: False)
    """

    print(f'========================================')
    print(f'Loading source weights from `{src_weight_path}` ...')
    sys.path.insert(0, CODE_PATH)
    with open(src_weight_path, 'rb') as f:
        model = pickle.load(f)
    sys.path.pop(0)
    print(f'Successfully loaded!')
    print(f'--------------------')

    z_space_dim = model['G'].z_dim
    label_size = model['G'].c_dim
    w_space_dim = model['G'].w_dim
    image_channels = model['G'].img_channels
    resolution = model['G'].img_resolution
    repeat_w = True

    print(f'Converting source weights (G) to target ...')
    G_vars = dict(model['G'].named_parameters())
    G_vars.update(dict(model['G'].named_buffers()))
    G = build_model(gan_type=GAN_TPYE,
                    module='generator',
                    resolution=resolution,
                    z_space_dim=z_space_dim,
                    w_space_dim=w_space_dim,
                    label_size=label_size,
                    repeat_w=repeat_w,
                    image_channels=image_channels)
    G_state_dict = G.state_dict()
    official_tf_to_pth_var_mapping = {}
    for name in G_vars.keys():
        for pattern, fn in G_PTH_TO_TF_VAR_MAPPING_PATTERN.items():
            match = re.fullmatch(pattern, name)
            if match:
                if fn is not None:
                    official_tf_to_pth_var_mapping[fn(*match.groups())] = name
                break
    for dst_var_name, tf_var_name in G.pth_to_tf_var_mapping.items():
        assert tf_var_name in official_tf_to_pth_var_mapping
        assert dst_var_name in G_state_dict
        src_var_name = official_tf_to_pth_var_mapping[tf_var_name]
        assert src_var_name in G_vars
        if verbose:
            print(f'    Converting `{src_var_name}` to `{dst_var_name}`.')
        var = G_vars[src_var_name].data
        if 'weight' in tf_var_name:
            if 'Conv0_up/weight' in tf_var_name:
                var = var.flip(2, 3)
            elif 'Skip' in tf_var_name:
                var = var.flip(2, 3)
        if 'bias' in tf_var_name:
            if 'mod_bias' in tf_var_name:
                var = var - 1
        if 'Const' in tf_var_name:
            var = var.unsqueeze(0)
        if 'noise' in tf_var_name and 'noise_' not in tf_var_name:
            var = var.unsqueeze(0).unsqueeze(0)
        G_state_dict[dst_var_name] = var
    print(f'Successfully converted!')
    print(f'--------------------')

    print(f'Converting source weights (Gs) to target ...')
    Gs_vars = dict(model['G_ema'].named_parameters())
    Gs_vars.update(dict(model['G_ema'].named_buffers()))
    Gs = build_model(gan_type=GAN_TPYE,
                     module='generator',
                     resolution=resolution,
                     z_space_dim=z_space_dim,
                     w_space_dim=w_space_dim,
                     label_size=label_size,
                     repeat_w=repeat_w,
                     image_channels=image_channels)
    Gs_state_dict = Gs.state_dict()
    official_tf_to_pth_var_mapping = {}
    for name in Gs_vars.keys():
        for pattern, fn in G_PTH_TO_TF_VAR_MAPPING_PATTERN.items():
            match = re.fullmatch(pattern, name)
            if match:
                if fn is not None:
                    official_tf_to_pth_var_mapping[fn(*match.groups())] = name
                break
    for dst_var_name, tf_var_name in Gs.pth_to_tf_var_mapping.items():
        assert tf_var_name in official_tf_to_pth_var_mapping
        assert dst_var_name in Gs_state_dict
        src_var_name = official_tf_to_pth_var_mapping[tf_var_name]
        assert src_var_name in Gs_vars
        if verbose:
            print(f'    Converting `{src_var_name}` to `{dst_var_name}`.')
        var = Gs_vars[src_var_name].data
        if 'weight' in tf_var_name:
            if 'Conv0_up/weight' in tf_var_name:
                var = var.flip(2, 3)
            elif 'Skip' in tf_var_name:
                var = var.flip(2, 3)
        if 'bias' in tf_var_name:
            if 'mod_bias' in tf_var_name:
                var = var - 1
        if 'Const' in tf_var_name:
            var = var.unsqueeze(0)
        if 'noise' in tf_var_name and 'noise_' not in tf_var_name:
            var = var.unsqueeze(0).unsqueeze(0)
        Gs_state_dict[dst_var_name] = var
    print(f'Successfully converted!')
    print(f'--------------------')

    print(f'Converting source weights (D) to target ...')
    D_vars = dict(model['D'].named_parameters())
    D_vars.update(dict(model['D'].named_buffers()))
    D = build_model(gan_type=GAN_TPYE,
                    module='discriminator',
                    resolution=resolution,
                    label_size=label_size,
                    image_channels=image_channels)
    D_state_dict = D.state_dict()
    official_tf_to_pth_var_mapping = {}
    for name in D_vars.keys():
        for pattern, fn in D_PTH_TO_TF_VAR_MAPPING_PATTERN.items():
            match = re.fullmatch(pattern, name)
            if match:
                if fn is not None:
                    official_tf_to_pth_var_mapping[fn(*match.groups())] = name
                break
    for dst_var_name, tf_var_name in D.pth_to_tf_var_mapping.items():
        assert tf_var_name in official_tf_to_pth_var_mapping
        assert dst_var_name in D_state_dict
        src_var_name = official_tf_to_pth_var_mapping[tf_var_name]
        assert src_var_name in D_vars
        if verbose:
            print(f'    Converting `{src_var_name}` to `{dst_var_name}`.')
        var = D_vars[src_var_name].data
        D_state_dict[dst_var_name] = var
    print(f'Successfully converted!')
    print(f'--------------------')

    print(f'Saving target weights to `{dst_weight_path}` ...')
    state_dict = {
        'generator': G_state_dict,
        'discriminator': D_state_dict,
        'generator_smooth': Gs_state_dict,
    }
    torch.save(state_dict, dst_weight_path)
    print(f'Successfully saved!')
    print(f'--------------------')

    # Start testing if needed.
    if test_num <= 0:
        warnings.warn(f'Skip testing the converted weights!')
        return

    if save_test_image:
        html = HtmlPageVisualizer(num_rows=test_num, num_cols=3)
        html.set_headers(['Index', 'Before Conversion', 'After Conversion'])
        for i in range(test_num):
            html.set_cell(i, 0, text=f'{i}')

    print(f'Testing conversion results ...')
    G.load_state_dict(G_state_dict)
    D.load_state_dict(D_state_dict)
    Gs.load_state_dict(Gs_state_dict)
    G.eval().cuda()
    D.eval().cuda()
    Gs.eval().cuda()
    model['G'].eval().cuda()
    model['D'].eval().cuda()
    model['G_ema'].eval().cuda()

    gs_distance = 0.0
    dg_distance = 0.0
    for i in tqdm(range(test_num)):
        # Test Gs(z).
        code = np.random.randn(1, z_space_dim)
        code = torch.from_numpy(code).type(torch.FloatTensor).cuda()
        if label_size:
            label_id = np.random.randint(label_size)
            label = np.zeros((1, label_size), np.float32)
            label[0, label_id] = 1.0
            label = torch.from_numpy(label).type(torch.FloatTensor).cuda()
        else:
            label_id = 0
            label = None

        src_output = model['G_ema'](code,
                                    label,
                                    truncation_psi=TRUNC_PSI,
                                    truncation_cutoff=TRUNC_LAYERS,
                                    noise_mode=NOISE_MODE)
        src_output = src_output.detach().cpu().numpy()
        dst_output = Gs(code,
                        label=label,
                        trunc_psi=TRUNC_PSI,
                        trunc_layers=TRUNC_LAYERS,
                        randomize_noise=RANDOMIZE_NOISE)['image']
        dst_output = dst_output.detach().cpu().numpy()
        distance = np.average(np.abs(src_output - dst_output))
        if verbose:
            print(f'    Test {i:03d}: Gs distance {distance:.6e}.')
        gs_distance += distance

        if save_test_image:
            html.set_cell(i, 1, image=postprocess_image(src_output)[0])
            html.set_cell(i, 2, image=postprocess_image(dst_output)[0])

        # Test D(G(z)).
        code = np.random.randn(1, z_space_dim)
        code = torch.from_numpy(code).type(torch.FloatTensor).cuda()
        if label_size:
            label_id = np.random.randint(label_size)
            label = np.zeros((1, label_size), np.float32)
            label[0, label_id] = 1.0
            label = torch.from_numpy(label).type(torch.FloatTensor).cuda()
        else:
            label_id = 0
            label = None
        src_image = model['G'](code,
                               label,
                               truncation_psi=TRUNC_PSI,
                               truncation_cutoff=TRUNC_LAYERS,
                               noise_mode=NOISE_MODE)
        src_output = model['D'](src_image, label)
        src_output = src_output.detach().cpu().numpy()
        dst_image = G(code,
                      label=label,
                      trunc_psi=TRUNC_PSI,
                      trunc_layers=TRUNC_LAYERS,
                      randomize_noise=RANDOMIZE_NOISE)['image']
        dst_output = D(dst_image, label)
        dst_output = dst_output.detach().cpu().numpy()
        distance = np.average(np.abs(src_output - dst_output))
        if verbose:
            print(f'    Test {i:03d}: D(G) distance {distance:.6e}.')
        dg_distance += distance

    print(f'Average Gs distance is {gs_distance / test_num:.6e}.')
    print(f'Average D(G) distance is {dg_distance / test_num:.6e}.')
    print(f'========================================')

    if save_test_image:
        html.save(f'{dst_weight_path}.conversion_test.html')
