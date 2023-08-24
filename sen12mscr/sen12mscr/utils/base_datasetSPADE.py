"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import numpy as np
import random

def lambda_proj_1(img): return 2*img-1
def lambda_proj_2(img): return 0.5*img+0.5


def lambda_AB(sample): return (
    sample['A']['S1'], sample['A']['S1 path'], sample['B']['S2'], sample['B']['S2 path'])


def lambda_A(sample): return (sample['S1'], sample['S1 path'])
def lambda_B(sample): return (sample['S2'], sample['S2 path'])


def lambda_only_AB(sample): return (sample['A']['S1'], sample['A']['S1 path'])
def lambda_only_A(sample): return sample['S1']
def lambda_only_B(sample): return sample['S2']

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size, size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=InterpolationMode.BICUBIC, use_hsv_aug=False, use_gray_aug=False, use_gaussian_blur=False, kernel_size=5, rescale_method='default'):
    transform_list = [
        transforms.Lambda(lambda img: torch.Tensor(img))
    ]
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(
            lambda img: __flip(img, params['flip'])))
    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(
            lambda img: __resize(img, w, h, method)))

    if use_hsv_aug:
        transform_list.append(transforms.RandomApply([_color_jitter], p=0.8))
    if use_gray_aug:
        transform_list.append(transforms.RandomGrayscale(p=0.2))
    if use_gaussian_blur:
        transform_list.append(transforms.GaussianBlur(
            kernel_size=kernel_size))

    if rescale_method == 'default' or 'clip' in rescale_method:
        transform_list.append(transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)


_s = 1
_color_jitter = transforms.ColorJitter(
    0.8 * _s, 0.8 * _s, 0.8 * _s, 0.2 * _s
)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=InterpolationMode.BICUBIC):
    return F.resize(img, (w, h), method)


def __make_power_2(img, base, method=InterpolationMode.BICUBIC):
    _, ow, oh = img.shape
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return F.resize(img, (w, h), method)


def __scale_width(img, target_width, method=InterpolationMode.BICUBIC):
    _, ow, oh = img.shape
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return F.resize(img, (w, h), method)


def __scale_shortside(img, target_width, method=InterpolationMode.BICUBIC):
    _, ow, oh = img.shape
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return F.resize(img, (nw, nh), method)


def __crop(img, pos, size):
    _, ow, oh = img.shape
    x1, y1 = pos
    tw = th = size
    return F.crop(img, x1, y1, tw, th)


def __flip(img, flip):
    if flip:
        return F.hflip(img)
    return img
