from sen12mscr.base_dataset import SEN12MSCR_A, SEN12MSCR_B
from sen12mscr.utils.base_dataset import BaseDataset
from sen12mscr.utils.utils import get_transform, copyconf, lambda_A, lambda_B
import random

class UnalignedDatasetCUT(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    """
    
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

        dataroot = opt.data_root
        split = 'train' if opt.isTrain else 'test_use_all'
        season = opt.sen12mscr_seaon
        s1_rescale_method = opt.s1_rescale_mtehod
        s2_rescale_method = opt.s2_rescale_mtehod
        s1_rgb_composite = opt.s1_rgb_composite
        # s1_transforms       = opt.s1_transforms
        # s2_transforms       = opt.s2_transforms
        btoA = self.opt.direction == 'BtoA'
                
        self.dataset_A = SEN12MSCR_A(dataroot, split=split, season=season, s1_rescale_method=s1_rescale_method,
                                     s2_rescale_method=s2_rescale_method, s1_rgb_composite=s1_rgb_composite, Lambda=lambda_A)
        self.dataset_B = SEN12MSCR_B(dataroot, split=split, season=season, s1_rescale_method=s1_rescale_method,
                                     s2_rescale_method=s2_rescale_method, s1_rgb_composite=s1_rgb_composite, Lambda=lambda_B)
        
        self.length = len(self.dataset_B)
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A, A_path = self.dataset_A[index]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.length
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.length - 1)
        B, B_path = self.dataset_B[index_B]
        
        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#       print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        if self.opt.isTrain and self.current_epoch == self.opt.n_epochs + 1:
            print("entering finetuning phase")
        modified_opt = copyconf(
            self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        
        transform_A = get_transform(modified_opt, use_hsv_aug=False, use_gray_aug=False,
                                    use_gaussian_blur=False, rescale_method=self.opt.s1_rescale_method)
        transform_B = get_transform(modified_opt, use_hsv_aug=self.opt.use_hsv_aug, use_gray_aug=self.opt.use_gray_aug,
                                    use_gaussian_blur=self.use_gaussian_blur, kernel_size=self.opt.kernel_size, rescale_method=self.opt.s2_rescale_method)
        
        A = transform_A(A)
        B = transform_B(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.length
