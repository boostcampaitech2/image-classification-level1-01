
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from efficientnet_pytorch import EfficientNet

from model.focal_loss import *
from data.dataset import *
from data.kfold import *

import copy
import unittest

cuda_flag = torch.cuda.is_available()
device = torch.device('cuda' if cuda_flag else 'cpu')

data_set = magv_dataset('/opt/ml/input/data/train/images', 
                        index=[i for i in range(2700*7)],
                        transforms='mask_transform', resize=312,
                        age_test_num=58, model=0)
                    
data_sampler = RandomSampler(data_set)
data_loader = DataLoader(data_set, 
                         batch_size=64,
                         shuffle=False, 
                         sampler=data_sampler, 
                         num_workers=4)

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=3).to(device)
opt = optim.Adam(model.parameters() , 1e-3)
ce_loss = nn.CrossEntropyLoss().to(device)


class ModelTest(unittest.TestCase): 
    
    def setUp(self) :
        
        self.device = device
        self.data_set = data_set
        self.data_loader = data_loader
        self.model = model
        self.opt = opt
        self.ce_loss = ce_loss

    # input type , dtype test
    def test_type(self) :
        
        sample_dict = self.data_set[0]
        sample_tensor = sample_dict['image']

        element_type = str(sample_tensor[0].dtype)
   
        self.assertIsInstance(sample_tensor , np.ndarray)
        self.assertEqual(element_type , 'float32')
        
    # input dimension test
    def test_input_dim(self) :
        
        sample_dict = self.data_set[0]
        sample_tensor = sample_dict['image']
            
        tensor_dim = sample_tensor.shape
            
        self.assertEqual(tensor_dim[0] , 3)
        self.assertEqual(tensor_dim[1] , 312)
        self.assertEqual(tensor_dim[2] , 312)
        
    # output dimension test
    def test_output_dim(self) :
        
        sample_dict = self.data_set[0]
        sample_tensor = sample_dict['image']
        
        sample_tensor = torch.tensor(sample_tensor).to(self.device)
        sample_tensor = sample_tensor.unsqueeze(0)
        
        output_tensor = self.model(sample_tensor)
        
        output_dim = output_tensor.shape
        
        self.assertEqual(len(output_dim) , 2)
        self.assertEqual(output_dim[1] , 3)
        
    # training test
    def test_train(self) :
        
        # parameters before training
        prev_param = [copy.deepcopy(m) for m in self.model.parameters()]
 
        for sample_dict in self.data_loader :
            break

        sample_tensor = sample_dict['image']
        sample_label = sample_dict['label']
        
        sample_in = sample_tensor.to(self.device)
        sample_label = sample_label.to(self.device)

        sample_out = self.model(sample_in)
        
        loss = self.ce_loss(sample_out , sample_label)
        
        loss.backward()
        self.opt.step()
        
        idx = 0
        
        # check if parameters are updated
        for m in self.model.parameters() :
            
            self.assertFalse(torch.equal(prev_param[idx] , m))
            idx += 1
   

if __name__ == '__main__' :

    unittest.main()
 