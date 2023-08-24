from data.base_dataset import BaseDataset
from sen12mscr.base_dataset import SEN12MSCR_AB
from sen12mscr.utils.utils import get_params, get_transform, lambda_AB


class AlignedPix2PixDataset(BaseDataset):
    """A dataset class for paired image dataset."""

    """
    opt is required to have following options
    
    use_sen12mscr       : store_true
    dataroot           : SEN12MS-CR dataroot
    sen12mscr_season    : SEN12MS-CR season
    isTrain             : model train phase
    opt.input_nc        : SAR channel dimension
    opt.output_nc       : EO channel dimension
    opt.BtoA            : EO to SAR    
    s1_rescale_method   : SAR rescale method
    s2_rescale_method   : EO rescale method
    s1_rgb_composite    : SAR 3rd channel method
    # s1_transforms       : SAR transforms method
    # s2_transforms       : EO transforms
    no_flip             : RandomHorizontalFlip
    preprocess          : resize_and_crop (default)
    load_size           : img load size
    crop_size           : img crop size
    use_hsv_aug         : RandomColorJitter (EO)
    use_gray_aug        : RandomGrayScale (EO)
    use_gaussian_blur   : GuassianBlur (EO)
    kernel_size         : GuassianBlur kernel_size (EO)
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.load_size = opt.load_size
        self.crop_size = opt.crop_size      
        assert self.load_size >= self.crop_size
        
        dataroot            = opt.dataroot
        split               = 'train' if opt.isTrain else 'test_use_all'
        season              = opt.sen12mscr_seaon
        s1_rescale_method   = opt.s1_rescale_mtehod
        s2_rescale_method   = opt.s2_rescale_mtehod
        s1_rgb_composite    = opt.s1_rgb_composite
        # s1_transforms       = opt.s1_transforms
        # s2_transforms       = opt.s2_transforms
        self.input_nc       = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc      = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.dataset = SEN12MSCR_AB(dataroot, split=split, season=season, s1_rescale_method=s1_rescale_method ,s2_rescale_method=s2_rescale_method, s1_rgb_composite=s1_rgb_composite, Lambda=lambda_AB)



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - SAR
            B (tensor) - - EO
            A_paths (str) - - SAR image paths
            B_paths (str) - - EO image paths
        """
        A, A_path, B, B_path = self.dataset[index]
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, self.load_size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), use_hsv_aug=False, use_gray_aug=False, use_gaussian_blur=False, rescale_method=self.opt.s1_rescale_method)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), use_hsv_aug=self.opt.use_hsv_aug, use_gray_aug=self.opt.use_gray_aug, use_gaussian_blur=self.opt.use_gaussian_blur, kernel_size=self.opts.kernel_size, rescale_method=self.opt.s2_rescale_method)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset)
