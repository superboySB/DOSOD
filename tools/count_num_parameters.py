import argparse
import torch


def parse_args():

    parser = argparse.ArgumentParser("Compute the number of parameters of a model")
    parser.add_argument('checkpoint', type=str, help='model checkpoint path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # load checkpoint
    model = torch.load(args.checkpoint, map_location='cpu')
    state_dict = model['state_dict']
    num_parameters = 0

    for k, v in state_dict.items():
        num_parameters += v.numel()

    print(f'num_parameters: {num_parameters} | {num_parameters / 1e6:.2f} MB')