import random
import numpy as np
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


def get_transform(opt, params=None,method=InterpolationMode.BICUBIC, use_hsv_aug=False, use_gray_aug=False, use_gaussian_blur=False, kernel_size=5, rescale_method='default'):
    transform_list = [
        transforms.Lambda(lambda img: torch.Tensor(img))
    ]
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(
                lambda img: __hflip(img, params['flip'])))
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base, method)))
                
    if use_hsv_aug:
        transform_list.append(transforms.RandomApply([_color_jitter], p=0.8))
    if use_gray_aug:
        transform_list.append(transforms.RandomGrayscale(p=0.2))
    if use_gaussian_blur:
        transform_list.append(transforms.GaussianBlur(kernel_size=kernel_size))

    if rescale_method == 'default' or 'clip' in rescale_method:
        transform_list.append(transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)

def get_params(opt, size):
    w, h = size, size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=InterpolationMode.BICUBIC):
    _, ow, oh = img.shape
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return F.resize(img, (w, h), method)


def __scale_width(img, target_size, method=InterpolationMode.BICUBIC):
    _, ow, oh = img.shape
    if ow == target_size:
        return img
    w = target_size
    h = int(target_size * oh / ow)
    return F.resize(img, (w, h), method)


def __crop(img, pos, size):
    _, ow, oh = img.shape
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return F.crop(img, x1, y1, tw, th)
    return img


def __hflip(img, flip):
    if flip:
        return F.hflip(img)
    return img


def __vflip(img, flip):
    if flip:
        return F.vflip(img)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
