# python3.7
"""A simple tool to synthesize images with pre-trained models."""

import os
import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch

from models import MODEL_ZOO
from models import build_generator
from utils.misc import bool_parser
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import postprocess_image
from utils.visualizer import save_image


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Synthesize images with pre-trained models.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/synthesis/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--num', type=int, default=100,
                        help='Number of samples to synthesize. '
                             '(default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--generate_html', type=bool_parser, default=True,
                        help='Whether to use HTML page to visualize the '
                             'synthesized results. (default: %(default)s)')
    parser.add_argument('--save_raw_synthesis', type=bool_parser, default=False,
                        help='Whether to save raw synthesis. '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if args.num <= 0:
        return
    if not args.save_raw_synthesis and not args.generate_html:
        return

    # Parse model configuration.
    if args.model_name not in MODEL_ZOO:
        raise SystemExit(f'Model `{args.model_name}` is not registered in '
                         f'`models/model_zoo.py`!')
    model_config = MODEL_ZOO[args.model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Get work directory and job name.
    if args.save_dir:
        work_dir = args.save_dir
    else:
        work_dir = os.path.join('work_dirs', 'synthesis')
    os.makedirs(work_dir, exist_ok=True)
    job_name = f'{args.model_name}_{args.num}'
    if args.save_raw_synthesis:
        os.makedirs(os.path.join(work_dir, job_name), exist_ok=True)

    # Build generation and get synthesis kwargs.
    print(f'Building generator for model `{args.model_name}` ...')
    generator = build_generator(**model_config)
    synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                            trunc_layers=args.trunc_layers,
                            randomize_noise=args.randomize_noise)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    print(f'Finish loading checkpoint.')

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Sample and synthesize.
    print(f'Synthesizing {args.num} samples ...')
    indices = list(range(args.num))
    if args.generate_html:
        html = HtmlPageVisualizer(grid_size=args.num)
    for batch_idx in tqdm(range(0, args.num, args.batch_size)):
        sub_indices = indices[batch_idx:batch_idx + args.batch_size]
        code = torch.randn(len(sub_indices), generator.z_space_dim).cuda()
        with torch.no_grad():
            images = generator(code, **synthesis_kwargs)['image']
            images = postprocess_image(images.detach().cpu().numpy())
        for sub_idx, image in zip(sub_indices, images):
            if args.save_raw_synthesis:
                save_path = os.path.join(
                    work_dir, job_name, f'{sub_idx:06d}.jpg')
                save_image(save_path, image)
            if args.generate_html:
                row_idx, col_idx = divmod(sub_idx, html.num_cols)
                html.set_cell(row_idx, col_idx, image=image,
                              text=f'Sample {sub_idx:06d}')
    if args.generate_html:
        html.save(os.path.join(work_dir, f'{job_name}.html'))
    print(f'Finish synthesizing {args.num} samples.')


if __name__ == '__main__':
    main()
