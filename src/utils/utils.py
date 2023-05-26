import torch
import numpy as np
import random
import collections
import json
import matplotlib.pyplot as plt
import sys
import os
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import math
import warnings


def print_loss_graph_from_file(file_name,out_file):
    file_in = open(file_name, 'r')
    lines = file_in.readlines()
    stats ={'train_loss':[],'val_loss':[],'best_loss':[]}
    for line in lines:
        stats_line = json.loads(line.replace('\n',''))
        stats['train_loss'].append(float(stats_line['train_loss']))
        stats['val_loss'].append(float(stats_line['val_loss']))
        stats['best_loss'].append(float(stats_line['best_loss']))
    file_in.close()
    plt.figure(0)
    plt.title('Loss')
    plt.plot(stats['train_loss'], label='train')
    plt.plot(stats['val_loss'], label='val')
    plt.plot(stats['best_loss'], label='best')
    plt.legend()
    plt.savefig(out_file)
    # plt.show()



def print_loss_graph_from_file_rl(file_name,out_file):
    file_in = open(file_name, 'r')
    lines = file_in.readlines()
    stats ={'train_rewards':[],'valid_rewards':[],'best_rewards':[]}
    for line in lines:
        stats_line = json.loads(line.replace('\n',''))
        stats['train_rewards'].append(float(stats_line['train_rewards']))
        stats['valid_rewards'].append(float(stats_line['valid_rewards']))
    file_in.close()
    best = stats['valid_rewards'][0]
    for r in stats['valid_rewards']:
        if r > best:
            best = r
        stats['best_rewards'].append(best)
    plt.figure(1)
    plt.title('Rewards')
    plt.plot(stats['train_rewards'], label='train')
    plt.plot(stats['valid_rewards'], label='val')
    plt.plot(stats['best_rewards'], label='best')
    plt.legend()
    plt.savefig(out_file)
    # plt.show()

def print_loss_graph_from_details_file_rl(mode,file_name,out_file):
    file_in = open(file_name, 'r')
    lines = file_in.readlines()
    stats ={'epoch':[],'train':[],'avr_train_loss':[],'avr_train_reward':[]}
    for line in lines:
        stats_line = json.loads(line.replace('\n',''))
        stats['avr_train_loss'].append(float(stats_line['avr_train_loss']))
        stats['avr_train_reward'].append(float(stats_line['avr_train_reward']))
    file_in.close()
    plt.figure(1)
    plt.title('Detailed rewards and loss RL')
    if mode == 'l':
        plt.plot(stats['avr_train_loss'], label='train loss')
    else:
        plt.plot(stats['avr_train_reward'], label='train reward')
    plt.legend()
    plt.savefig(out_file)
    # plt.show()

def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device, non_blocking=True)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")



def pairwise_distance_matrix(x, y):
    H, d = x.shape
    W, _ = y.shape

    x = x.unsqueeze(1).expand(H, W, d)
    y = y.unsqueeze(0).expand(H, W, d)

    return (x - y).pow(2).sum(dim=-1)


def label_clustering(labels, keys=None):
    """
    labels.shape = B, H
    keys.shape = K, H
    """
    if keys is None:
        keys = labels.unique(dim=0)

    dist_mat = pairwise_distance_matrix(labels, keys) # B, K
    membership = dist_mat.argmin(dim=-1) # B

    return membership

def retrieve_hints_from_solution(solution_board,filename):
    data_type = '-test'
    solution_board = solution_board.reshape(9,9)
    solutions = np.load(f'data/{filename}/{filename}_sol{data_type}.npy',allow_pickle=True).item()
    hints = np.load(f'data/{filename}/{filename}{data_type}.npy',allow_pickle=True).item()
    index = [i for i in range(len(solutions)) if (solutions[i] == solution_board).all()]
    return hints[index[0]]


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

