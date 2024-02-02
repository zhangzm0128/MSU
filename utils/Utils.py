import time
import numpy as np
import torch


def format_runtime(time_gap):
    m, s = divmod(time_gap, 60)
    h, m = divmod(m, 60)
    runtime_str = ''
    if h != 0:
        runtime_str = runtime_str + '{}h'.format(int(h))
        runtime_str = runtime_str + '{}m'.format(int(m))
        runtime_str = runtime_str + '{}s'.format(int(s))
    elif m != 0:
        runtime_str = runtime_str + '{}m'.format(int(m))
        runtime_str = runtime_str + '{}s'.format(int(s))
    else:
        runtime_str = runtime_str + '{:.4}s'.format(s)
    return runtime_str

def binarization_mask(mask_sigmoid):
    mask_binary = mask_sigmoid.clone()
    mask_binary[mask_binary > 0.5] = 1
    mask_binary[mask_binary <= 0.5] = 0
    return mask_binary

def sum_mask_point(mask_binary):
    return torch.sum(mask_binary, dim=1)