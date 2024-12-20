import os
import argparse

import torch
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser("Reparameterize DOSOD")
    parser.add_argument('--model', help='model checkpoints to reparameterize')
    parser.add_argument('--out-dir', help='output checkpoints')
    parser.add_argument(
        '--text-embed',
        help='text embeddings to be reparameterized')

    args = parser.parse_args()
    return args


def convert_head(scale, bias, text_embed):
    N, D = text_embed.shape
    weight = (text_embed * scale.exp()).view(N, D, 1, 1)
    bias = torch.ones(N) * bias
    return weight, bias


def reparameterize_head(state_dict, embeds):

    cls_layers = [
        'bbox_head.head_module.cls_contrasts.0',
        'bbox_head.head_module.cls_contrasts.1',
        'bbox_head.head_module.cls_contrasts.2'
    ]

    for i in range(3):
        scale = state_dict[cls_layers[i] + '.logit_scale']
        bias = state_dict[cls_layers[i] + '.bias']
        weight, bias = convert_head(scale, bias, embeds)
        state_dict[cls_layers[i] + '.conv.weight'] = weight
        state_dict[cls_layers[i] + '.conv.bias'] = bias
        del state_dict[cls_layers[i] + '.bias']
        del state_dict[cls_layers[i] + '.logit_scale']
    return state_dict


def main():

    args = parse_args()

    # load checkpoint
    model = torch.load(args.model, map_location='cpu')
    state_dict = model['state_dict']

    # load embeddings
    embeddings = torch.from_numpy(np.load(args.text_embed))

    # remove text encoder and text adaptor
    keys = list(state_dict.keys())
    keys = [x for x in keys if "backbone_text" not in x and 'text_mlp' not in x]

    state_dict_wo_text = {x: state_dict[x] for x in keys}
    print("removing text encoder")

    state_dict_wo_text = reparameterize_head(state_dict_wo_text, embeddings)
    print("reparameterizing head")

    model['state_dict'] = state_dict_wo_text

    model_name = os.path.basename(args.model)
    model_name = model_name.replace('.pth', f'_rep.pth')
    torch.save(model, os.path.join(args.out_dir, model_name))


if __name__ == "__main__":
    main()
