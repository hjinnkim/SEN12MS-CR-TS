"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from data.base_dataset import BaseDataset
from sen12mscr.utils.base_datasetSPADE import get_params, get_transform, lambda_AB
from sen12mscr.base_dataset import SEN12MSCR_AB
from torchvision.transforms import InterpolationMode
from PIL import Image

# From Pix2PixDataset class of SPADE
def modify_commandline_options(parser, is_train):
    parser.add_argument('--no_pairing_check', action='store_true',
                        help='If specified, skip sanity check of correct label-image file pairing')
    return parser

class AlignedSPADEDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = modify_commandline_options(parser, is_train)
        parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.load_size = opt.load_size
        self.crop_size = opt.crop_size
        
        dataroot            = opt.dataroot
        split               = 'train' if opt.isTrain else 'test_use_all'
        season              = opt.sen12mscr_seaon
        s1_rescale_method   = opt.s1_rescale_mtehod
        s2_rescale_method   = opt.s2_rescale_mtehod
        s1_rgb_composite    = opt.s1_rgb_composite
        

        self.dataset = SEN12MSCR_AB(dataroot, split=split, season=season, s1_rescale_method=s1_rescale_method,
                                    s2_rescale_method=s2_rescale_method, s1_rgb_composite=s1_rgb_composite, Lambda=lambda_AB)
        
        size = len(self.dataset)
        self.dataset_size = size

    def __getitem__(self, index):
        A, A_path, B, B_path = self.dataset[index]
        params = get_params(self.opt, self.load_size)
                
        transform_label = get_transform(
            self.opt, params, method=InterpolationMode.NEAREST, use_hsv_aug=False, use_gray_aug=False, use_gaussian_blur=False, kernel_size=self.opt.kernel_size, rescale_method='default')
        label_tensor = transform_label(B)
        
        # input image (real images)
        transform_image = get_transform(self.opt, params, use_hsv_aug=self.opt.use_hsv_aug, use_gray_aug=self.opt.use_gray_aug, use_gaussian_blur=self.use_gaussian_blur, kernel_size=self.opts.kernel_size, rescale_method=self.opt.s2_rescale_method)
        image_tensor = transform_image(A)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': A_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
