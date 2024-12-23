"""
    Configs for training
    Written by Bobby Wang
"""

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        default='./data/images',
        type=str,
        help='Directory path of image data')
    parser.add_argument(
        '--mask_path',
        default='./data/labels',
        type=str,
        help='Directory path of label data')
    parser.add_argument(
        '--gt_path',
        default='./data/class.xlsx',
        type=str,
        help='Directory path to save results')
    parser.add_argument(
        '--pretrain',
        default='None',
        type=str,
        help='Path for pretrained model.')
    parser.add_argument(
        '--phase',
        default='train_test',
        type=str,
        help='Phase of train or test')
    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        type=float,
        help='Initial learning rate')
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--frozen_epochs',
        default=10,
        type=int,
        help='Number of frozen epochs to run')
    parser.add_argument(
        '--input_size',
        default=(32, 224, 224),
        type=tuple,
        help='Input size of depth, height, width')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet')
    parser.add_argument(
        '--model_depth',
        default=10,
        type=int,
        help='Depth of convolutional neural network, choices=10, 18, 34, 50, 101, 152')
    parser.add_argument(
        '--save_intervals',
        default=5,
        type=int,
        help='Interation for saving model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_opts()
    print(args)