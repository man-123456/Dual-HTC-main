import torch
import random
import numpy as np
from collections import defaultdict


def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# seed_torch(3)
# print('Set seed to 3.')

def get_hierarchy_info(label_cpt):
    """
    :param label_cpt: the path of the label_cpt file
    :return: hiera: Dict{str -> Set[str]}, the parent-child relationship of labels
    :return: _label_dict: Dict{str -> int}, the label to id mapping
    :return: r_hiera: Dict{str -> str}, the child-parent relationship of labels
    :return: label_depth: Dict{str -> int}, the depth of each label
    """
    hiera = defaultdict(set)
    _label_dict = {}
    with open(label_cpt) as f:
        _label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in _label_dict:
                    _label_dict[i] = len(_label_dict) - 1
                hiera[line[0]].add(i)
        _label_dict.pop('Root')
    r_hiera = {}
    for i in hiera:
        for j in list(hiera[i]):
            r_hiera[j] = i

    def _loop(a):
        if r_hiera[a] != 'Root':
            return [a, ] + _loop(r_hiera[a])
        else:
            return [a]

    label_depth = {}
    for i in _label_dict:
        label_depth[i] = len(_loop(i))

    return hiera, _label_dict, r_hiera, label_depth
