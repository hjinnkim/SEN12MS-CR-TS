import random
import numpy as np
from argparse import Namespace

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

lambda_proj_1 = lambda img: 2*img-1
lambda_proj_2 = lambda img: 0.5*img+0.5

lambda_AB = lambda sample: (sample['A']['S1'], sample['A']['S1 path'], sample['B']['S2'], sample['B']['S2 path'])
lambda_A = lambda sample: (sample['S1'], sample['S1 path'])
lambda_B = lambda sample: (sample['S2'], sample['S2 path'])

lambda_only_AB = lambda sample: (sample['A']['S1'], sample['A']['S1 path'])
lambda_only_A = lambda sample: sample['S1']
lambda_only_B = lambda sample: sample['S2']

_s=1
_color_jitter = transforms.ColorJitter(
    0.8 * _s, 0.8 * _s, 0.8 * _s, 0.2 * _s
)


transform_Tensor = transforms.Compose([
    transforms.Lambda(lambda img : torch.Tensor(img)),
])

# NICE-GAN
def get_base_transforms(load_size: int=286, img_size: int=256, rescale_method='default', use_hsv_aug=False, use_gray_aug=False, use_gaussian_blur=False, kernel_size=5):
    
    _train_transforms=[
        transforms.Lambda(lambda img : torch.Tensor(img)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((load_size, load_size)),
        transforms.RandomCrop(img_size),
    ]
    _test_transforms=[
        transforms.Lambda(lambda img : torch.Tensor(img)),
        transforms.Resize((img_size, img_size)),
    ]
    if use_hsv_aug:
        _train_transforms+=[transforms.RandomApply([_color_jitter], p=0.8)]
    if use_gray_aug:
        _train_transforms+=[transforms.RandomGrayscale(p=0.2)]
    if use_gaussian_blur:
        _train_transforms+=[transforms.GaussianBlur(kernel_size=kernel_size)]
        
    if rescale_method=='default' or 'clip' in rescale_method:
        _train_transforms+=[transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        _test_transforms+=[transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    
    train_transforms = transforms.Compose(_train_transforms)
    test_transforms = transforms.Compose(_test_transforms)
    
    return train_transforms, test_transforms

# Pix2Pix, CycleGAN


def get_transform(opt, params=None, grayscale=False, use_hsv_aug=False, method=InterpolationMode.BICUBIC, rescale_method='linear'):
    transform_list = [
        transforms.Lambda(lambda img: torch.Tensor(img)[None, :])
    ]
    if grayscale:
        transform_list.append(transforms.Grayscale(3))
    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(
                lambda img: __hflip(img, params['flip'])))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
        # if rescale_method == 'default' or 'clip' in rescale_method:
        #     transform_list.append(transforms.Lambda(lambda img: (img-img.min())/(img.max()-img.min())))
        
        
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base=4, method=method)))
        
    if use_hsv_aug:
        transform_list.append(transforms.RandomApply([_color_jitter], p=0.8))

    if rescale_method == 'linear':
        transform_list.append(transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform_list.append(transforms.Lambda(lambda img: img.squeeze()))
    
    
    return transforms.Compose(transform_list)

def get_params(opt, size):
    w, h = size, size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def __make_power_2(img, base, method=InterpolationMode.BICUBIC):
    _, _, ow, oh = img.shape
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return F.resize(img, (w, h), method)


def __scale_width(img, target_size, crop_size, method=InterpolationMode.BICUBIC):
    _, _, ow, oh = img.shape
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return F.resize(img, (w, h), method)


def __crop(img, pos, size):
    _, _, ow, oh = img.shape
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return F.crop(img, x1, y1, tw, th)
    return img


def __hflip(img, flip):
    if flip:
        return F.hflip(img)
    return img

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf
