import torch
import torchvision.transforms as transforms

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

