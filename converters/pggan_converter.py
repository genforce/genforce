# python3.7
"""Converts PGGAN model weights from TensorFlow to PyTorch.

The models can be trained through OR released by the repository:

https://github.com/tkarras/progressive_growing_of_gans
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# pylint: disable=wrong-import-position
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import torch
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from models import build_model
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import postprocess_image
# pylint: enable=wrong-import-position

__all__ = ['convert_pggan_weight']

GAN_TPYE = 'pggan'
OFFICIAL_CODE_DIR = 'pggan_official'
BASE_DIR = os.path.dirname(os.path.relpath(__file__))
CODE_PATH = os.path.join(BASE_DIR, OFFICIAL_CODE_DIR)


def convert_pggan_weight(tf_weight_path,
                         pth_weight_path,
                         test_num=10,
                         save_test_image=False,
                         verbose=False):
    """Converts the pre-trained PGGAN weights.

    Args:
        tf_weight_path: Path to the TensorFlow model to load weights from.
        pth_weight_path: Path to the PyTorch model to save converted weights.
        test_num: Number of samples used to test the conversion. (default: 10)
        save_test_image: Whether to save the test images. (default: False)
        verbose: Whether to print verbose log message. (default: False)
    """
    sess = tf.compat.v1.InteractiveSession()

    print(f'========================================')
    print(f'Loading TensorFlow weights from `{tf_weight_path}` ...')
    sys.path.insert(0, CODE_PATH)
    with open(tf_weight_path, 'rb') as f:
        G, D, Gs = pickle.load(f)
    sys.path.pop(0)
    print(f'Successfully loaded!')
    print(f'--------------------')

    z_space_dim = G.input_shapes[0][1]
    label_size = G.input_shapes[1][1]
    image_channels = G.output_shape[1]
    resolution = G.output_shape[2]

    print(f'Converting TensorFlow weights (G) to PyTorch version ...')
    G_vars = dict(G.__getstate__()['variables'])
    G_pth = build_model(gan_type=GAN_TPYE,
                        module='generator',
                        resolution=resolution,
                        z_space_dim=z_space_dim,
                        label_size=label_size,
                        image_channels=image_channels)
    G_state_dict = G_pth.state_dict()
    for pth_var_name, tf_var_name in G_pth.pth_to_tf_var_mapping.items():
        assert tf_var_name in G_vars
        assert pth_var_name in G_state_dict
        if verbose:
            print(f'    Converting `{tf_var_name}` to `{pth_var_name}`.')
        var = torch.from_numpy(np.array(G_vars[tf_var_name]))
        if 'weight' in tf_var_name:
            if 'Dense' in tf_var_name:
                var = var.view(var.shape[0], -1, G_pth.init_res, G_pth.init_res)
                var = var.permute(1, 0, 2, 3).flip(2, 3)
            else:
                var = var.permute(3, 2, 0, 1)
        G_state_dict[pth_var_name] = var
    print(f'Successfully converted!')
    print(f'--------------------')

    print(f'Converting TensorFlow weights (Gs) to PyTorch version ...')
    Gs_vars = dict(Gs.__getstate__()['variables'])
    Gs_pth = build_model(gan_type=GAN_TPYE,
                         module='generator',
                         resolution=resolution,
                         z_space_dim=z_space_dim,
                         label_size=label_size,
                         image_channels=image_channels)
    Gs_state_dict = Gs_pth.state_dict()
    for pth_var_name, tf_var_name in Gs_pth.pth_to_tf_var_mapping.items():
        assert tf_var_name in Gs_vars
        assert pth_var_name in Gs_state_dict
        if verbose:
            print(f'    Converting `{tf_var_name}` to `{pth_var_name}`.')
        var = torch.from_numpy(np.array(Gs_vars[tf_var_name]))
        if 'weight' in tf_var_name:
            if 'Dense' in tf_var_name:
                var = var.view(
                    var.shape[0], -1, Gs_pth.init_res, Gs_pth.init_res)
                var = var.permute(1, 0, 2, 3).flip(2, 3)
            else:
                var = var.permute(3, 2, 0, 1)
        Gs_state_dict[pth_var_name] = var
    print(f'Successfully converted!')
    print(f'--------------------')

    print(f'Converting TensorFlow weights (D) to PyTorch version ...')
    D_vars = dict(D.__getstate__()['variables'])
    D_pth = build_model(gan_type=GAN_TPYE,
                        module='discriminator',
                        resolution=resolution,
                        label_size=label_size,
                        image_channels=image_channels)
    D_state_dict = D_pth.state_dict()
    for pth_var_name, tf_var_name in D_pth.pth_to_tf_var_mapping.items():
        assert tf_var_name in D_vars
        assert pth_var_name in D_state_dict
        if verbose:
            print(f'  Converting `{tf_var_name}` to `{pth_var_name}`.')
        var = torch.from_numpy(np.array(D_vars[tf_var_name]))
        if 'weight' in tf_var_name:
            if 'Dense' in tf_var_name:
                var = var.permute(1, 0)
            else:
                var = var.permute(3, 2, 0, 1)
        D_state_dict[pth_var_name] = var
    print(f'Successfully converted!')
    print(f'--------------------')

    print(f'Saving PyTorch weights to `{pth_weight_path}` ...')
    state_dict = {
        'generator': G_state_dict,
        'discriminator': D_state_dict,
        'generator_smooth': Gs_state_dict,
    }
    torch.save(state_dict, pth_weight_path)
    print(f'Successfully saved!')
    print(f'--------------------')

    # Start testing if needed.
    if test_num <= 0 or not tf.test.is_built_with_cuda():
        warnings.warn(f'Skip testing the converted weights!')
        sess.close()
        return

    if save_test_image:
        html = HtmlPageVisualizer(num_rows=test_num, num_cols=3)
        html.set_headers(['Index', 'Before Conversion', 'After Conversion'])
        for i in range(test_num):
            html.set_cell(i, 0, text=f'{i}')

    print(f'Testing conversion results ...')
    G_pth.load_state_dict(G_state_dict)
    D_pth.load_state_dict(D_state_dict)
    Gs_pth.load_state_dict(Gs_state_dict)
    G_pth.eval().cuda()
    D_pth.eval().cuda()
    Gs_pth.eval().cuda()

    gs_distance = 0.0
    dg_distance = 0.0
    for i in tqdm(range(test_num)):
        # Test Gs(z).
        code = np.random.randn(1, z_space_dim)
        pth_code = torch.from_numpy(code).type(torch.FloatTensor).cuda()
        label = np.zeros((1, label_size), np.float32)
        if label_size:
            label_id = np.random.randint(label_size)
            label[0, label_id] = 1.0
            pth_label = torch.from_numpy(label).type(torch.FloatTensor).cuda()
        else:
            label_id = 0
            pth_label = None
        tf_output = Gs.run(code, label)
        pth_output = Gs_pth(pth_code, label=pth_label)['image']
        pth_output = pth_output.detach().cpu().numpy()
        distance = np.average(np.abs(tf_output - pth_output))
        if verbose:
            print(f'    Test {i:03d}: Gs distance {distance:.6e}.')
        gs_distance += distance

        if save_test_image:
            html.set_cell(i, 1, image=postprocess_image(tf_output)[0])
            html.set_cell(i, 2, image=postprocess_image(pth_output)[0])

        # Test D(G(z)).
        code = np.random.randn(1, z_space_dim)
        pth_code = torch.from_numpy(code).type(torch.FloatTensor).cuda()
        label = np.zeros((1, label_size), np.float32)
        if label_size:
            label_id = np.random.randint(label_size)
            label[0, label_id] = 1.0
            pth_label = torch.from_numpy(label).type(torch.FloatTensor).cuda()
        else:
            label_id = 0
            pth_label = None
        tf_image = G.run(code, label)
        tf_output = D.run(tf_image)
        pth_image = G_pth(pth_code, label=pth_label)['image']
        pth_output = D_pth(pth_image)
        pth_output = pth_output.detach().cpu().numpy()
        distance = np.average(np.abs(tf_output[0] - pth_output[:, :1]))
        if label_size:
            distance += np.average(np.abs(tf_output[1] - pth_output[:, 1:]))
        if verbose:
            print(f'    Test {i:03d}: D(G) distance {distance:.6e}.')
        dg_distance += distance

    print(f'Average Gs distance is {gs_distance / test_num:.6e}.')
    print(f'Average D(G) distance is {dg_distance / test_num:.6e}.')
    print(f'========================================')

    if save_test_image:
        html.save(f'{pth_weight_path}.conversion_test.html')

    sess.close()
