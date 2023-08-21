from sen12mscr.base_dataset import SEN12MSCR_A, SEN12MSCR_B
from sen12mscr.utils.utils import lambda_only_A, lambda_only_B, get_base_transforms
from torch.utils.data import DataLoader

def get_all_data_loaders(opts):
    batch_size          = opts.batch_size
    num_workers         = opts.num_workers
    data_root           = opts.data_root
    new_size            = opts.new_size
    crop_size           = opts.crop_size
    use_hsv_aug         = opts.use_hsv_aug
    use_gray_aug        = opts.use_gray_aug
    use_gaussian_blur   = opts.gaussian_blur
    kernel_size         = opts.kernel_size    
    s1_rescale_method   = opts.s1_rescale_method
    s2_rescale_method   = opts.s2_rescale_method
    
    transforms_A_train, transforms_A_test = get_base_transforms(
        load_size=new_size, img_size=crop_size, rescale_method=s1_rescale_method, use_hsv_aug=False, use_gray_aug=False, use_gaussian_blur=False, kernel_size=kernel_size)
    transforms_B_train, transforms_B_test = get_base_transforms(
        load_size=new_size, img_size=crop_size, rescale_method=s2_rescale_method, use_hsv_aug=use_hsv_aug, use_gray_aug=use_gray_aug, use_gaussian_blur=use_gaussian_blur, kernel_size=kernel_size)
    
    train_set_A = SEN12MSCR_A(data_root, split='train', season=opts.sen12mscr_season, s1_rescale_method=opts.s1_rescale_method,
                              s1_rgb_composite=opts.s1_rgb_composite, s1_transforms=transforms_A_train, Lambda=lambda_only_A)
    
    train_set_B = SEN12MSCR_B(data_root, split='train', season=opts.sen12mscr_season,
                              s2_rescale_method=opts.s2_rescale_method, s2_transforms=transforms_B_train, Lambda=lambda_only_B)
    
    
    test_set_A = SEN12MSCR_A(data_root, split='test_use_all', season=opts.sen12mscr_season, s1_rescale_method=opts.s1_rescale_method,
                             s1_rgb_composite=opts.s1_rgb_composite, s1_transforms=transforms_A_test, Lambda=lambda_only_A)
    
    test_set_B = SEN12MSCR_B(data_root, split='test_use_all', season=opts.sen12mscr_season,
                             s2_rescale_method=opts.s2_rescale_method, s2_transforms=transforms_B_test, Lambda=lambda_only_B)

    train_loader_a = DataLoader(train_set_A, batch_size=batch_size,
                                shuffle=True, drop_last=True, num_workers=num_workers)
    train_loader_b = DataLoader(train_set_B, batch_size=batch_size,
                                shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader_a = DataLoader(test_set_A, batch_size=batch_size,
                               shuffle=False, drop_last=True, num_workers=num_workers)
    test_loader_b = DataLoader(train_set_A, batch_size=batch_size,
                                shuffle=False, drop_last=True, num_workers=num_workers)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b