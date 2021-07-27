"""Script to convert officially released models to match this repository."""

import argparse

from converters import convert_pggan_weight
from converters import convert_stylegan_weight
from converters import convert_stylegan2_weight
from converters import convert_stylegan2ada_tf_weight
from converters import convert_stylegan2ada_pth_weight


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Convert pre-trained models.')
    parser.add_argument('model_type', type=str,
                        choices=['pggan', 'stylegan', 'stylegan2',
                                 'stylegan2ada_tf', 'stylegan2ada_pth'],
                        help='Type of the model to convert')
    parser.add_argument('--source_model_path', type=str, required=True,
                        help='Path to load the model for conversion.')
    parser.add_argument('--target_model_path', type=str, default=None,
                        help='Path to save the converted model. If not '
                             'specified, the model will be saved to the same '
                             'directory of the source model.')
    parser.add_argument('--test_num', type=int, default=10,
                        help='Number of test samples used to check the '
                             'precision of the converted model. (default: 10)')
    parser.add_argument('--save_test_image', action='store_true',
                        help='Whether to save the test image. (default: False)')
    parser.add_argument('--verbose_log', action='store_true',
                        help='Whether to print verbose log. (default: False)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if args.target_model_path is None:
        args.target_model_path = args.source_model_path.replace('.pkl', '.pth')

    if args.model_type == 'pggan':
        convert_pggan_weight(tf_weight_path=args.source_model_path,
                             pth_weight_path=args.target_model_path,
                             test_num=args.test_num,
                             save_test_image=args.save_test_image,
                             verbose=args.verbose_log)
    elif args.model_type == 'stylegan':
        convert_stylegan_weight(tf_weight_path=args.source_model_path,
                                pth_weight_path=args.target_model_path,
                                test_num=args.test_num,
                                save_test_image=args.save_test_image,
                                verbose=args.verbose_log)
    elif args.model_type == 'stylegan2':
        convert_stylegan2_weight(tf_weight_path=args.source_model_path,
                                 pth_weight_path=args.target_model_path,
                                 test_num=args.test_num,
                                 save_test_image=args.save_test_image,
                                 verbose=args.verbose_log)
    elif args.model_type == 'stylegan2ada_tf':
        convert_stylegan2ada_tf_weight(tf_weight_path=args.source_model_path,
                                       pth_weight_path=args.target_model_path,
                                       test_num=args.test_num,
                                       save_test_image=args.save_test_image,
                                       verbose=args.verbose_log)
    elif args.model_type == 'stylegan2ada_pth':
        convert_stylegan2ada_pth_weight(src_weight_path=args.source_model_path,
                                        dst_weight_path=args.target_model_path,
                                        test_num=args.test_num,
                                        save_test_image=args.save_test_image,
                                        verbose=args.verbose_log)
    else:
        raise NotImplementedError(f'Model type `{args.model_type}` is not '
                                  f'supported!')


if __name__ == '__main__':
    main()
