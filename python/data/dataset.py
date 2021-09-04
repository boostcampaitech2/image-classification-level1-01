import torch
from torch.utils.data import Dataset
import cv2
from data.transforms import *
from data.transforms_sub import *
from data.prepare_data import *

class magv_dataset(Dataset):
    # transforms
    def __init__(
        self, data_path, index=None,
        transforms=None,resize=None, 
        age_test_num=None, model=None,sub=0):
        
        if sub==0:
            self.index=index
            self.mask_data = prepare_mask_data(data_path,subject=sub)
            self.age_gender_data = prepare_age_gender_data(self.mask_data,age_test_num)
            self.data = concat_data(self.mask_data, self.age_gender_data).iloc[self.index]
            self.transforms = magv_transforms(mode=transforms, resize=resize)
        else:
            self.index=index
            self.mask_data = prepare_mask_data(data_path,subject=sub)
            self.age_gender_data = prepare_age_gender_data(self.mask_data,age_test_num)
            self.data = concat_data_sub(self.mask_data, self.age_gender_data).iloc[self.index]
            self.transforms = magv_transforms_sub(mode=transforms, resize=resize)
        
        self.model=model+1 #mask, gender, age 판별하는 변수
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = int(self.data.iloc[index,self.model])
        img_path = self.data.iloc[index,0]
        
        img = cv2.imread(img_path)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image=image)
        image = image.transpose((2,0,1))
        sample = {'image': image, 'label': label}
        
        return sample


class dataset_test(Dataset):
    def __init__(self, data, transforms=None, resize=312, sub=0):
        self.data = data
        
        if sub==0:
            self.path = '/opt/ml/input/data/eval/new_imgs'
            self.transforms = magv_transforms(mode=transforms, resize=resize)
        else:
            self.path = '/opt/ml/input/data/eval/images'
            self.transforms = magv_transforms_sub(mode=transforms, resize=resize)
        #/new_imgs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx,0]
        img = cv2.imread(os.path.join(self.path, img_path))
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image=image)
        image = image.transpose((2,0,1))
        sample = {'image': image}
        
        return sample
        
class dataset_valid(Dataset):
    def __init__(self, data, transforms=None, resize=312):
        self.data = data
        self.transforms=magv_transforms(
            mode=transforms, resize=resize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label_mask = int(self.data.iloc[idx, 1])
        label_age = int(self.data.iloc[idx, 2])
        label_gender = int(self.data.iloc[idx, 3])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=img)
        image = image.transpose((2, 0, 1))
        sample = {
            'image':image,
            'label_mask':label_mask,
            'label_age':label_age,
            'label_gender':label_gender
        }

        return sample