import os
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from abc import ABC, abstractmethod

from datetime import datetime
to_date   = lambda string: datetime.strptime(string, '%Y-%m-%d')
S1_LAUNCH = to_date('2014-04-03')

import rasterio
from torch.utils.data import Dataset

################SEN12MS-CR Stastics##################
# SAR statistics
# VV / VH / (VV/VH) or (VV+VH)/2
S1_min      = {
    'add'   : {
        0.0: np.array([[[-25.0]], [[-25.0]], [[-25.0]]], dtype=np.float32),
        1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
    },
    'div'   : {
        0.0: np.array([[[-25.0]], [[-25.0]], [[0.0]]], dtype=np.float32),
        1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
    }
}
S1_max      = {
    'add'   : {
        0.0: np.array([[[0.0]], [[0.0]], [[0.0]]], dtype=np.float32),
        1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
    },
    'div'   : {
        0.0: np.array([[[0.0]], [[0.0]], [[2.0]]], dtype=np.float32),
        1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
    }
}
S1_avg      = {
    True:{
        'add'   : {
            0.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
        },
        'div'   : {
            0.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
        }
    }, 
    False:{
        'add'   : {
            0.0: np.array([[[-11.3838]], [[-18.1206]], [[-14.7522]]], dtype=np.float32),
            1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
        },
        'div'   : {
            0.0: np.array([[[-11.3838]], [[-18.1206]], [[0.6216]]], dtype=np.float32),
            1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
        }
    }
}
S1_std      = {
    True:{
        'add'   : {
            0.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
        },
        'div'   : {
            0.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
        }
    }, 
    False:{
        'add'   : {
            0.0: np.array([[[4.4542]], [[4.9335]], [[4.4379]]], dtype=np.float32),
            1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
        },
        'div'   : {
            0.0: np.array([[[4.4542]], [[4.9335]], [[0.1645]]], dtype=np.float32),
            1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
            2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
        }
    }
}

# EO statistics
# B4 / B3 / B2
S2_min      = { 
    0.0: np.array([[[0.0]], [[0.0]], [[0.0]]], dtype=np.float32),
    1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
    2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
}
S2_max      = { 
    0.0: np.array([[[4000]], [[4000]], [[4000]]], dtype=np.float32),
    1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
    2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
}
S2_avg      = {
    True:{
        0.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
    }, 
    False:{
        0.0: np.array([[[1028.1276]], [[1069.8891]], [[1167.1272]]], dtype=np.float32),
        1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
    }, 
}
S2_std      = {
    True:{
        0.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
    }, 
    False:{
        0.0: np.array([[[753.9400]], [[560.4429]], [[541.1479]]], dtype=np.float32),
        1.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32),
        2.0: np.array([[[None]], [[None]], [[None]]], dtype=np.float32)
    }, 
}

#####################################################

# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif

def read_img(tif):
    return np.nan_to_num(tif.read().astype(np.float32))

def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def cut_percent(img, percent):
    cut_value = np.percentile(img, [percent, 100.-percent], axis=(1,2), interpolation='nearest')
    img = np.clip(img, cut_value[0][:, None, None], cut_value[1][:, None, None])
    return img

def process_MS(img, method, percent=0.0, per_image=False):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    if method=='base':
        img = np.clip(img, 0, 10000)
    if method=='linear':
        img = np.clip(img, 0, 10000)
        if per_image:
            intensity_min = np.min(img, axis=(1,2))[:, None, None]
            intensity_max = np.max(img, axis=(1,2))[:, None, None]
        else:
            intensity_min = S2_min[percent]
            intensity_max = S2_max[percent]
            img = np.clip(img, intensity_min, intensity_max)
        img = rescale(img, intensity_min, intensity_max)
    if method=='norm':
        img = np.clip(img, 0, 10000)
        if per_image:
            intensity_min = np.min(img, axis=(1,2))[:, None, None]
            intensity_max = np.max(img, axis=(1,2))[:, None, None]
            img = rescale(img, intensity_min, intensity_max)
        else:
            intensity_min = S2_min[percent]
            intensity_max = S2_max[percent]
            img = np.clip(img, intensity_min, intensity_max)
        img = (img-S2_avg[per_image][percent])/S2_std[per_image][percent]
    return img

def process_SAR(img, method, composite, percent=0.0, per_image=False):
    if method=='default':
        dB_min, dB_max = -25, 0                            # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                 # intensity clipping to a global unified SAR dB range
        img = rescale(img, dB_min, dB_max)                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                              (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    if method=='base':
        img = np.clip(img, -50, 0)
        img = S1_RGB_Composite(img, composite)
    if method=='linear':
        img = np.clip(img, -50, 0)
        img = S1_RGB_Composite(img, composite)
        if per_image:
            dB_min = np.min(img, axis=(1,2))[:, None, None]
            dB_max = np.max(img, axis=(1,2))[:, None, None]
        else:
            dB_min = S1_min[composite][percent]
            dB_max = S1_max[composite][percent]
            img = np.clip(img, dB_min, dB_max)
        img = rescale(img, dB_min, dB_max)  
    if method=='norm':
        img = np.clip(img, -50, 0)
        img = S1_RGB_Composite(img, composite)
        if per_image:
            dB_min = np.min(img, axis=(1,2))[:, None, None]
            dB_max = np.max(img, axis=(1,2))[:, None, None]
            img = rescale(img, dB_min, dB_max)  
        else:
            dB_min = S1_min[composite][percent]
            dB_max = S1_max[composite][percent]
            img = np.clip(img, dB_min, dB_max)
        img = (img-S1_avg[per_image][composite][percent])/S1_std[per_image][composite][percent]
    return img

def S1_RGB_Composite(img, method):   
    if method=='add':
        img = np.stack((img[0], img[1], (img[0]+img[1])/2.0), axis=0)
    if method=='div':
        img = np.stack((img[0], img[1], np.clip(np.divide(img[0], img[1], out=np.zeros_like(img[0]), where=(img[1] != 0)), 0.0, 2.0)), axis=0)
    
    return img

class SEN12MSCRBase(Dataset, ABC): # A : SAR / B : EO
    def __init__(self, root, split="all", region='all', season='all', s1_rescale_method='default', s2_rescale_method='default', cut_percent=0.0, s1_rgb_composite='add', per_image=False, s1_transforms=None, s2_transforms=None, Lambda=None):

        self.root_dir = root   # set root directory which contains all ROI
        self.region   = region # region according to which the ROI are selected # TODO: currently only supporting 'all'
        
        _season = season.split(',')
        seasons = ['spring', 'summer', 'fall', 'winter']
        if 'all' in _season:
            self.season = 'all'
        else:
            assert not any([season not in seasons for season in _season]), "Input season must be either assigned as all, spring, summer, fall, winter, or comma separated e.g., \'spring,summer\' (no space allowed, e.g., \'spring, fall\') !"
            self.season   = _season
        self.s1_rgb_composite = s1_rgb_composite
        self.Lambda = Lambda
        self.s1_transforms = s1_transforms
        self.s2_transforms = s2_transforms
        self.ROI      = {'ROIs1158': ['106'],
                         'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
                         'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
                         'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']}
        
        # define splits conform with SEN12MS-CR-TS
        self.splits         = {}
        self.splits['train']= ['ROIs1970_fall_s1/s1_3', 'ROIs1970_fall_s1/s1_22', 'ROIs1970_fall_s1/s1_148', 'ROIs1970_fall_s1/s1_107', 'ROIs1970_fall_s1/s1_1', 'ROIs1970_fall_s1/s1_114', 
                               'ROIs1970_fall_s1/s1_135', 'ROIs1970_fall_s1/s1_40', 'ROIs1970_fall_s1/s1_42', 'ROIs1970_fall_s1/s1_31', 'ROIs1970_fall_s1/s1_149', 'ROIs1970_fall_s1/s1_64', 
                               'ROIs1970_fall_s1/s1_28', 'ROIs1970_fall_s1/s1_144', 'ROIs1970_fall_s1/s1_57', 'ROIs1970_fall_s1/s1_35', 'ROIs1970_fall_s1/s1_133', 'ROIs1970_fall_s1/s1_30', 
                               'ROIs1970_fall_s1/s1_134', 'ROIs1970_fall_s1/s1_141', 'ROIs1970_fall_s1/s1_112', 'ROIs1970_fall_s1/s1_116', 'ROIs1970_fall_s1/s1_37', 'ROIs1970_fall_s1/s1_26', 
                               'ROIs1970_fall_s1/s1_77', 'ROIs1970_fall_s1/s1_100', 'ROIs1970_fall_s1/s1_83', 'ROIs1970_fall_s1/s1_71', 'ROIs1970_fall_s1/s1_93', 'ROIs1970_fall_s1/s1_119', 
                               'ROIs1970_fall_s1/s1_104', 'ROIs1970_fall_s1/s1_136', 'ROIs1970_fall_s1/s1_6', 'ROIs1970_fall_s1/s1_41', 'ROIs1970_fall_s1/s1_125', 'ROIs1970_fall_s1/s1_91', 
                               'ROIs1970_fall_s1/s1_131', 'ROIs1970_fall_s1/s1_120', 'ROIs1970_fall_s1/s1_110', 'ROIs1970_fall_s1/s1_19', 'ROIs1970_fall_s1/s1_14', 'ROIs1970_fall_s1/s1_81', 
                               'ROIs1970_fall_s1/s1_39', 'ROIs1970_fall_s1/s1_109', 'ROIs1970_fall_s1/s1_33', 'ROIs1970_fall_s1/s1_88', 'ROIs1970_fall_s1/s1_11', 'ROIs1970_fall_s1/s1_128', 
                               'ROIs1970_fall_s1/s1_142', 'ROIs1970_fall_s1/s1_122', 'ROIs1970_fall_s1/s1_4', 'ROIs1970_fall_s1/s1_27', 'ROIs1970_fall_s1/s1_147', 'ROIs1970_fall_s1/s1_85', 
                               'ROIs1970_fall_s1/s1_82', 'ROIs1970_fall_s1/s1_105', 'ROIs1158_spring_s1/s1_9', 'ROIs1158_spring_s1/s1_1', 'ROIs1158_spring_s1/s1_124', 'ROIs1158_spring_s1/s1_40', 
                               'ROIs1158_spring_s1/s1_101', 'ROIs1158_spring_s1/s1_21', 'ROIs1158_spring_s1/s1_134', 'ROIs1158_spring_s1/s1_145', 'ROIs1158_spring_s1/s1_141', 'ROIs1158_spring_s1/s1_66', 
                               'ROIs1158_spring_s1/s1_8', 'ROIs1158_spring_s1/s1_26', 'ROIs1158_spring_s1/s1_77', 'ROIs1158_spring_s1/s1_113', 'ROIs1158_spring_s1/s1_100', 
                               'ROIs1158_spring_s1/s1_117', 'ROIs1158_spring_s1/s1_119', 'ROIs1158_spring_s1/s1_6', 'ROIs1158_spring_s1/s1_58', 'ROIs1158_spring_s1/s1_120', 'ROIs1158_spring_s1/s1_110', 
                               'ROIs1158_spring_s1/s1_126', 'ROIs1158_spring_s1/s1_115', 'ROIs1158_spring_s1/s1_121', 'ROIs1158_spring_s1/s1_39', 'ROIs1158_spring_s1/s1_109', 'ROIs1158_spring_s1/s1_63', 
                               'ROIs1158_spring_s1/s1_75', 'ROIs1158_spring_s1/s1_132', 'ROIs1158_spring_s1/s1_128', 'ROIs1158_spring_s1/s1_142', 'ROIs1158_spring_s1/s1_15', 'ROIs1158_spring_s1/s1_45', 
                               'ROIs1158_spring_s1/s1_97', 'ROIs1158_spring_s1/s1_147', 'ROIs1868_summer_s1/s1_90', 'ROIs1868_summer_s1/s1_87', 'ROIs1868_summer_s1/s1_25', 'ROIs1868_summer_s1/s1_124', 
                               'ROIs1868_summer_s1/s1_114', 'ROIs1868_summer_s1/s1_135', 'ROIs1868_summer_s1/s1_40', 'ROIs1868_summer_s1/s1_101', 'ROIs1868_summer_s1/s1_42', 
                               'ROIs1868_summer_s1/s1_31', 'ROIs1868_summer_s1/s1_36', 'ROIs1868_summer_s1/s1_139', 'ROIs1868_summer_s1/s1_56', 'ROIs1868_summer_s1/s1_133', 'ROIs1868_summer_s1/s1_55', 
                               'ROIs1868_summer_s1/s1_43', 'ROIs1868_summer_s1/s1_113', 'ROIs1868_summer_s1/s1_100', 'ROIs1868_summer_s1/s1_76', 'ROIs1868_summer_s1/s1_123', 'ROIs1868_summer_s1/s1_143', 
                               'ROIs1868_summer_s1/s1_93', 'ROIs1868_summer_s1/s1_125', 'ROIs1868_summer_s1/s1_89', 'ROIs1868_summer_s1/s1_120', 'ROIs1868_summer_s1/s1_126', 'ROIs1868_summer_s1/s1_72', 
                               'ROIs1868_summer_s1/s1_115', 'ROIs1868_summer_s1/s1_121', 'ROIs1868_summer_s1/s1_146', 'ROIs1868_summer_s1/s1_140', 'ROIs1868_summer_s1/s1_95', 
                               'ROIs1868_summer_s1/s1_102', 'ROIs1868_summer_s1/s1_7', 'ROIs1868_summer_s1/s1_11', 'ROIs1868_summer_s1/s1_132', 'ROIs1868_summer_s1/s1_15', 'ROIs1868_summer_s1/s1_137', 
                               'ROIs1868_summer_s1/s1_4', 'ROIs1868_summer_s1/s1_27', 'ROIs1868_summer_s1/s1_147', 'ROIs1868_summer_s1/s1_86', 'ROIs1868_summer_s1/s1_47', 'ROIs2017_winter_s1/s1_68', 
                               'ROIs2017_winter_s1/s1_25', 'ROIs2017_winter_s1/s1_62', 'ROIs2017_winter_s1/s1_135', 'ROIs2017_winter_s1/s1_42', 'ROIs2017_winter_s1/s1_64', 'ROIs2017_winter_s1/s1_21', 
                               'ROIs2017_winter_s1/s1_55', 'ROIs2017_winter_s1/s1_112', 'ROIs2017_winter_s1/s1_116', 'ROIs2017_winter_s1/s1_8', 'ROIs2017_winter_s1/s1_59', 'ROIs2017_winter_s1/s1_49', 
                               'ROIs2017_winter_s1/s1_104',  'ROIs2017_winter_s1/s1_81', 'ROIs2017_winter_s1/s1_146', 'ROIs2017_winter_s1/s1_75', 
                               'ROIs2017_winter_s1/s1_94', 'ROIs2017_winter_s1/s1_102', 'ROIs2017_winter_s1/s1_61', 'ROIs2017_winter_s1/s1_47']
        self.splits['val']  = ['ROIs2017_winter_s1/s1_22', 'ROIs1868_summer_s1/s1_19', 'ROIs1970_fall_s1/s1_65', 'ROIs1158_spring_s1/s1_17', 'ROIs2017_winter_s1/s1_107', 
                               'ROIs1868_summer_s1/s1_80', 'ROIs1868_summer_s1/s1_127', 'ROIs2017_winter_s1/s1_130', 'ROIs1868_summer_s1/s1_17', 'ROIs2017_winter_s1/s1_84'] 
        self.splits['test'] = ['ROIs1158_spring_s1/s1_106', 'ROIs1158_spring_s1/s1_123', 'ROIs1158_spring_s1/s1_140', 'ROIs1158_spring_s1/s1_31', 'ROIs1158_spring_s1/s1_44', 
                               'ROIs1868_summer_s1/s1_119', 'ROIs1868_summer_s1/s1_73', 'ROIs1970_fall_s1/s1_139', 'ROIs2017_winter_s1/s1_108', 'ROIs2017_winter_s1/s1_63']

        self.splits["all"]  = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.splits["test_use_all"] = self.splits["test"] + self.splits["val"]
        self.split = split
        
        assert split in ['all', 'train', 'val', 'test', 'test_use_all'], "Input dataset must be either assigned as all, train, test, val or test_use_all!"

        self.modalities         = ["S1", "S2"]
        self.s1_method          = s1_rescale_method
        self.s2_method          = s2_rescale_method
        self.cut_percent        = cut_percent
        self.per_image          = per_image
        
    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:

        path/to/your/SEN12MSCR/directory:
            ├───ROIs1158_spring_s1
            |   ├─s1_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s1_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2
            |   ├─s2_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2_cloudy
            |   ├─s2_cloudy_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_cloudy_1_p407.tif
            |   |   |...
            |    ...
            ...

        Note: Please arrange the dataset in a format as e.g. provided by the script dl_data.sh.
        """)
        
    @abstractmethod
    def get_paths(self):
        pass
    
    @abstractmethod
    def __getitem__(self, index):
        pass
    
    @abstractmethod
    def __len__(self):
        return 0


class SEN12MSCR_AB(SEN12MSCRBase, ABC): # A : SAR / B : EO
    def __init__(self, root, split="all", region='all', season='all', s1_rescale_method='default', s2_rescale_method='default', cut_percent=0.0, s1_rgb_composite='mean', per_image=False, s1_transforms=None, s2_transforms=None, Lambda=None):
        SEN12MSCRBase.__init__(self, root=root, split=split, region=region, season=season, s1_rescale_method=s1_rescale_method, s2_rescale_method=s2_rescale_method, cut_percent=cut_percent, s1_rgb_composite=s1_rgb_composite, per_image=per_image, s1_transforms=s1_transforms, s2_transforms=s2_transforms, Lambda=Lambda)
        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning if no data has been found
        if not self.n_samples: self.throw_warn()

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region} for split of season {self.season}')

        paths = []
        seeds_S1 = natsorted([s1dir for s1dir in os.listdir(self.root_dir) if "_s1" in s1dir])
        if not self.season == 'all':
            seasons = self.season
            seeds_S1 = natsorted([sldir for sldir in seeds_S1 if any([season in sldir for season in seasons])])
        
        for seed in tqdm(seeds_S1):
            rois_S1 = natsorted(os.listdir(os.path.join(self.root_dir, seed)))
            for roi in rois_S1:
                roi_dir  = os.path.join(self.root_dir, seed, roi)
                paths_S1        = natsorted([os.path.join(roi_dir, s1patch) for s1patch in os.listdir(roi_dir)])
                paths_S2        = [patch.replace('/s1', '/s2').replace('_s1', '_s2') for patch in paths_S1]

                for pdx, _ in enumerate(paths_S1):
                    if not all([os.path.isfile(paths_S1[pdx]), os.path.isfile(paths_S2[pdx])]): continue
                    if not any([split_roi == '/'.join(paths_S1[pdx].split('/')[-3:-1]) for split_roi in self.splits[self.split]]): continue
                    sample = {"S1":         paths_S1[pdx],
                              "S2":         paths_S2[pdx]}
                    paths.append(sample)
        return paths

    def __getitem__(self, pdx):  # get the triplet of patch with ID pdx
        s1_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S1']))
        s2_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S2']))
        coord           = list(s2_tif.bounds)
        s1              = cut_percent(read_img(s1_tif), self.cut_percent) if self.cut_percent > 0.0 else read_img(s1_tif)
        s2              = cut_percent(read_img(s2_tif)[[3,2,1],:,:], self.cut_percent) if self.cut_percent > 0.0 else read_img(s2_tif)[[3,2,1],:,:]
        
        s1              = process_SAR(s1, self.s1_method, self.s1_rgb_composite, self.cut_percent, self.per_image)
        if self.s1_transforms is not None:
            s1 = self.s1_transforms(s1)
        if self.s2_transforms is not None:
            s2 = self.transforms(s2)
        sample = {
                'A': {
                    'S1': s1,
                    'S1 path': os.path.join(self.root_dir, self.paths[pdx]['S1']),
                    'coord': coord,
                        },
                'B': {
                    'S2': process_MS(s2, self.s2_method, self.cut_percent, self.per_image),
                    'S2 path': os.path.join(self.root_dir, self.paths[pdx]['S2']),
                    'coord': coord,
                        },
                    }
        if self.Lambda is not None:
            sample = self.Lambda(sample)
        return sample
    
    def __len__(self):
        # length of generated list
        return self.n_samples
    
class SEN12MSCR_A(SEN12MSCRBase, ABC): # A : SAR / B : EO
    def __init__(self, root, split="all", region='all', season='all', s1_rescale_method='default', s2_rescale_method='default', cut_percent=0.0, s1_rgb_composite='mean', per_image=False, s1_transforms=None, s2_transforms=None, Lambda=None):
        SEN12MSCRBase.__init__(self, root=root, split=split, region=region, season=season, s1_rescale_method=s1_rescale_method, s2_rescale_method=s2_rescale_method, cut_percent=cut_percent, s1_rgb_composite=s1_rgb_composite, per_image=per_image, s1_transforms=s1_transforms, s2_transforms=None, Lambda=Lambda)
        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning if no data has been found
        if not self.n_samples: self.throw_warn()

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region} for split of season {self.season}')

        paths = []
        seeds_S1 = natsorted([s1dir for s1dir in os.listdir(self.root_dir) if "_s1" in s1dir])
        if not self.season == 'all':
            seasons = self.season
            seeds_S1 = natsorted([sldir for sldir in seeds_S1 if any([season in sldir for season in seasons])])
        
        for seed in tqdm(seeds_S1):
            rois_S1 = natsorted(os.listdir(os.path.join(self.root_dir, seed)))
            for roi in rois_S1:
                roi_dir  = os.path.join(self.root_dir, seed, roi)
                paths_S1        = natsorted([os.path.join(roi_dir, s1patch) for s1patch in os.listdir(roi_dir)])
                paths_S2        = [patch.replace('/s1', '/s2').replace('_s1', '_s2') for patch in paths_S1]

                for pdx, _ in enumerate(paths_S1):
                    if not all([os.path.isfile(paths_S1[pdx]), os.path.isfile(paths_S2[pdx])]): continue
                    if not any([split_roi == '/'.join(paths_S1[pdx].split('/')[-3:-1]) for split_roi in self.splits[self.split]]): continue
                    sample = {"S1":         paths_S1[pdx]}
                    paths.append(sample)
        return paths

    def __getitem__(self, pdx):  # get the triplet of patch with ID pdx
        s1_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S1']))
        coord           = list(s1_tif.bounds)
        s1              = cut_percent(read_img(s1_tif), self.cut_percent) if self.cut_percent > 0.0 else read_img(s1_tif)
        s1              = process_SAR(s1, self.s1_method, self.s1_rgb_composite, self.cut_percent, self.per_image)
        if self.s1_transforms is not None:
            s1 = self.s1_transforms(s1)
        sample = {
                'S1': s1,
                'S1 path': os.path.join(self.root_dir, self.paths[pdx]['S1']),
                'coord': coord,
                    }
        if self.Lambda is not None:
            sample = self.Lambda(sample)
        return sample
    
    def __len__(self):
        # length of generated list
        return self.n_samples
    
class SEN12MSCR_B(SEN12MSCRBase, ABC): # A : SAR / B : EO
    def __init__(self, root, split="all", region='all', season='all', s1_rescale_method='default', s2_rescale_method='default', cut_percent=0.0, s1_rgb_composite='mean', per_image=False, s1_transforms=None, s2_transforms=None, Lambda=None):
        SEN12MSCRBase.__init__(self, root=root, split=split, region=region, season=season, s1_rescale_method=s1_rescale_method, s2_rescale_method=s2_rescale_method, cut_percent=cut_percent, s1_rgb_composite=None, s1_transforms=None, per_image=per_image, s2_transforms=s2_transforms, Lambda=Lambda)
        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning if no data has been found
        if not self.n_samples: self.throw_warn()

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region} for split of season {self.season}')

        paths = []
        seeds_S1 = natsorted([s1dir for s1dir in os.listdir(self.root_dir) if "_s1" in s1dir])
        if not self.season == 'all':
            seasons = self.season
            seeds_S1 = natsorted([sldir for sldir in seeds_S1 if any([season in sldir for season in seasons])])
        
        for seed in tqdm(seeds_S1):
            rois_S1 = natsorted(os.listdir(os.path.join(self.root_dir, seed)))
            for roi in rois_S1:
                roi_dir  = os.path.join(self.root_dir, seed, roi)
                paths_S1        = natsorted([os.path.join(roi_dir, s1patch) for s1patch in os.listdir(roi_dir)])
                paths_S2        = [patch.replace('/s1', '/s2').replace('_s1', '_s2') for patch in paths_S1]

                for pdx, _ in enumerate(paths_S1):
                    if not all([os.path.isfile(paths_S1[pdx]), os.path.isfile(paths_S2[pdx])]): continue
                    if not any([split_roi == '/'.join(paths_S1[pdx].split('/')[-3:-1]) for split_roi in self.splits[self.split]]): continue
                    sample = {"S2":         paths_S2[pdx]}
                    paths.append(sample)
        return paths

    def __getitem__(self, pdx):  # get the triplet of patch with ID pdx
        s2_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S2']))
        coord           = list(s2_tif.bounds)
        s2              = cut_percent(read_img(s2_tif)[[3,2,1],:,:], self.cut_percent) if self.cut_percent > 0.0 else read_img(s2_tif)[[3,2,1],:,:]
        if self.s2_transforms is not None:
            s2 = self.s2_transforms(s2)
        sample = {
                'S2': process_MS(s2, self.s2_method, self.cut_percent, self.per_image),
                'S2 path': os.path.join(self.root_dir, self.paths[pdx]['S2']),
                'coord': coord,
                    }
        if self.Lambda is not None:
            sample = self.Lambda(sample)
        return sample
    
    def __len__(self):
        # length of generated list
        return self.n_samples