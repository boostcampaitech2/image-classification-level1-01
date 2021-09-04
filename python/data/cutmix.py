import numpy as np
import random
import torch
def rand_bbox(size,lam):
    # size : [Batch_size, Channel, Width, Height]
    width = size[2] #이미지 width
    height = size[3] #이미지 height
    cut_ratio = np.sqrt(1. - lam) # 패티 크기의 비율 정하기
    cut_width = np.int(width * cut_ratio) #패치의 너비
    cut_height = np.int(height * cut_ratio) #패치의 높이
    
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    
    # 세로로 잘라질 수 있도록..! (가로 이미지는 다 살린다) -> 패치 부분 좌표값 추출
    bbx1 = 0 #np.clip(cx - cut_width // 2.0, width)
    bby1 = np.clip(cy-cut_height//2,0,height)
    bbx2 = width #np.clip(cx + cut_width // 2.0, width)
    bby2 = np.clip(cy+cut_height//2,0,height)
    
    return bbx1,bby1,bbx2,bby2

class CutMix(object):
    def __init__(self, beta, cutmix_prob) -> None:
        super().__init__()
        self.beta = beta
        self.cutmix_prob = cutmix_prob
    
    def forward(self, images, labels):
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(images.size()[0]).cuda()
        label_1 = labels #원본 이미지 label
        label_2 = labels[rand_index] # 패치 이미지 label
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1-((bbx2-bbx1) * (bby2-bby1) / (images.size()[-1] * images.size()[-2]))
        
        return {'lam' : lam, 'image': images, 'label_1' : label_1, 'label_2' : label_2}