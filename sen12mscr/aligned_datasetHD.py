from sen12mscr.base_dataset import SEN12MSCR_AB
from sen12mscr.utils.base_datasetHD import BaseDataset
from sen12mscr.utils.utilsHD import get_params, get_transform, lambda_AB


class AlignedDatasetHD(BaseDataset):
    """A dataset class for paired image dataset."""

    """
    opt is required to have following options
    
    use_sen12mscr       : store_true
    dataroot            : SEN12MS-CR data root
    sen12mscr_season    : SEN12MS-CR season
    isTrain             : model train phase
    input_nc            : SAR channel dimension
    output_nc           : EO channel dimension
    label_nc            : set 0
    no_instance         : set True
    load_features       : set False
    s1_rescale_method   : SAR rescale method
    s2_rescale_method   : EO rescale method
    s1_rgb_composite    : SAR 3rd channel method
    # s1_transforms       : SAR transforms method
    # s2_transforms       : EO transforms
    no_flip             : RandomHorizontalFlip
    resize_or_crop      : resize_and_crop (default)
    loadSize            : img load size
    fineSize            : img crop size
    use_hsv_aug         : RandomColorJitter (EO)
    use_gray_aug        : RandomGrayScale (EO)
    use_gaussian_blur   : GuassianBlur (EO)
    kernel_size         : GuassianBlur kernel_size (EO)
    """
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase
        
        split = 'train' if opt.isTrain else 'test_use_all'
        season = opt.sen12mscr_seaon
        self.loadSize = opt.load_size
        self.cropSize = opt.crop_size
        s1_rescale_method = opt.s1_rescale_mtehod
        s2_rescale_method = opt.s2_rescale_mtehod
        s1_rgb_composite = opt.s1_rgb_composite
        # s1_transforms       = opt.s1_transforms
        # s2_transforms       = opt.s2_transforms

        self.dataset = SEN12MSCR_AB(self.root, split=split, season=season, s1_rescale_method=s1_rescale_method,
                                    s2_rescale_method=s2_rescale_method, s1_rgb_composite=s1_rgb_composite, Lambda=lambda_AB)
        
        self.dataset_size = len(self.dataset)
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - EO
            B (tensor) - - SAR
            A_paths (str) - - EO image paths
            B_paths (str) - - SAR image paths
        """
        inst_tensor = feat_tensor = 0
        B, B_path, A, A_path = self.dataset[index]
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, self.loadSize)
        B_transform = get_transform(self.opt, transform_params, use_hsv_aug=False, use_gray_aug=False, use_gaussian_blur=False, rescale_method=self.opt.s1_rescale_method)
        A_transform = get_transform(self.opt, transform_params, use_hsv_aug=self.opt.use_hsv_aug, use_gray_aug=self.opt.use_gray_aug, use_gaussian_blur=self.use_gaussian_blur, kernel_size=self.opt.kernel_size, rescale_method=self.opt.s2_rescale_method)

        A = A_transform(A)
        B = B_transform(B)

        return {'label': A, 'inst': inst_tensor, 'image': B, 'feat': feat_tensor, 'path': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset_size) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
