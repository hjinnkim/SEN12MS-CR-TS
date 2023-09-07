from numpy import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from sen12mscr.sen12mscr.base_dataset import SEN12MSCR_AB, SEN12MSCR_A, SEN12MSCR_B
from sen12mscr.utils.utils import lambda_AB, lambda_A, lambda_B

def hflip(img, flip):
    if flip:
        return F.hflip(img)
    return img

def get_params(flip_p):
    flip = random.random() > 1-flip_p
    return flip

class SEN12MSCRABBase(Dataset):
    def __init__(self,
                 data_root,
                 split,
                 season,
                 s1_rescale_method,
                 s2_rescale_method,
                 s1_rgb_composite,
                 per_image=False,
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self._dataset = SEN12MSCR_AB(data_root, split=split, season=season, s1_rescale_method=s1_rescale_method,  s2_rescale_method=s2_rescale_method, s1_rgb_composite=s1_rgb_composite, per_image=per_image, Lambda=lambda_AB)
        self._length = len(self._dataset)
        self.flip_p = flip_p
        self.to_tensor = transforms.Lambda(lambda img: torch.Tensor(img))
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        _flip = get_params(self.flip_p)
        S1, S1_path, S2, S2_path = self._dataset[i]
        S1 = torch.permute(self.normalize(hflip(self.to_tensor(S1), _flip)), (1, 2, 0))
        S2 = torch.permute(self.normalize(hflip(self.to_tensor(S2), _flip)), (1, 2, 0))
        example['S1']=S1
        example['S2']=S2
        example['S1_path']=S1_path
        example['S2_path']=S2_path
        
        return example

class SEN12MSCRABase(Dataset):
    def __init__(self,
                 data_root,
                 split,
                 season,
                 s1_rescale_method,
                 s2_rescale_method,
                 s1_rgb_composite,
                 per_image=False,
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self._dataset = SEN12MSCR_A(data_root, split=split, season=season, s1_rescale_method=s1_rescale_method,  s2_rescale_method=s2_rescale_method, s1_rgb_composite=s1_rgb_composite, per_image=per_image, Lambda=lambda_A)
        self._length = len(self._dataset)
        self.flip_p = flip_p
        self.transpose = transforms.Compose([transforms.Lambda(lambda img: torch.Tensor(img)),
                                             transforms.RandomHorizontalFlip(p=flip_p),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        # self.to_tensor = transforms.Lambda(lambda img: torch.Tensor(img))
        # self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        # self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        S1, S1_path = self._dataset[i]
        S1 = torch.permute(self.transpose(S1), (1, 2, 0))
        example['S1']=S1
        example['S1_path']=S1_path
        
        return example

class SEN12MSCRBBase(Dataset):
    def __init__(self,
                 data_root,
                 split,
                 season,
                 s1_rescale_method,
                 s2_rescale_method,
                 s1_rgb_composite,
                 per_image=False,
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self._dataset = SEN12MSCR_B(data_root, split=split, season=season, s1_rescale_method=s1_rescale_method,  s2_rescale_method=s2_rescale_method, s1_rgb_composite=s1_rgb_composite, per_image=per_image, Lambda=lambda_A)
        self._length = len(self._dataset)
        self.flip_p = flip_p
        self.to_tensor = transforms.Lambda(lambda img: torch.Tensor(img))
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        S2, S2_path = self._dataset[i]
        S2 = torch.permute(self.transpose(S2), (1, 2, 0))
        example['S2']=S2
        example['S1_path']=S2_path
        
        return example
    
    
class SEN12MSCRABTrain(SEN12MSCRABBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="", split='train', season="", s1_rescale_method='linear', s2_rescale_method='linear', s1_rgb_composite='add')
        

class SEN12MSCRABValidation(SEN12MSCRABBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_root="", split='test_use_all', season="", s1_rescale_method='linear', s2_rescale_method='linear', s1_rgb_composite='add', flip_p=flip_p)
        
    
class SEN12MSCRATrain(SEN12MSCRABBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="", split='train', season="", s1_rescale_method='linear', s2_rescale_method='linear', s1_rgb_composite='add')
        

class SEN12MSCRAValidation(SEN12MSCRABBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_root="", split='test_use_all', season="", s1_rescale_method='linear', s2_rescale_method='linear', s1_rgb_composite='add', flip_p=flip_p)
        
    
class SEN12MSCRATrain(SEN12MSCRABBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="", split='train', season="", s1_rescale_method='linear', s2_rescale_method='linear', s1_rgb_composite='add')
        

class SEN12MSCRAValidation(SEN12MSCRABBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_root="", split='test_use_all', season="", s1_rescale_method='linear', s2_rescale_method='linear', s1_rgb_composite='add', flip_p=flip_p)
# 
# 
# class LSUNChurchesValidation(LSUNBase):
#     def __init__(self, flip_p=0., **kwargs):
#         super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
#                          flip_p=flip_p, **kwargs)
# 
# 
# class LSUNBedroomsTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)
# 
# 
# class LSUNBedroomsValidation(LSUNBase):
#     def __init__(self, flip_p=0.0, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
#                          flip_p=flip_p, **kwargs)
# 
# 
# class LSUNCatsTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)
# 
# 
# class LSUNCatsValidation(LSUNBase):
#     def __init__(self, flip_p=0., **kwargs):
#         super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
#                          flip_p=flip_p, **kwargs)