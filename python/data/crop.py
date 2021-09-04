import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2, tqdm


def crop_train_imgs(crop):
    print('=' * 19 + f' {crop} crop train imgs ' + '=' * 19)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'	
    img_path = '/opt/ml/input/data/train/images'
    new_img_dir = '/opt/ml/input/data/train/new_imgs'
    mtcnn = MTCNN(keep_all=True, device=device)
    cnt = 0
    
    if os.path.exists(new_img_dir):
        if crop == 'Stay':
            print('Already exists!')
            return
    else:
        os.mkdir(new_img_dir)

    for paths in tqdm.tqdm(os.listdir(img_path)):
        if paths[0] == '.': continue
        
        sub_dir = os.path.join(img_path, paths)
        
        for imgs in os.listdir(sub_dir):
            if imgs[0] == '.': continue
            img_dir = os.path.join(sub_dir, imgs)
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            boxes, probs = mtcnn.detect(img)
            
            if not isinstance(boxes, np.ndarray):
                img = img[100:400, 50:350, :]
            
            else:
                xmin = int(boxes[0, 0]) - 15
                ymin = int(boxes[0, 1]) - 15
                xmax = int(boxes[0, 2]) + 15
                ymax = int(boxes[0, 3]) + 15
                
                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                if xmax > 384: xmax = 384
                if ymax > 512: ymax = 512
                
                img = img[ymin:ymax, xmin:xmax, :]
            
            tmp = os.path.join(new_img_dir, paths)
            if not os.path.exists(tmp):
                os.makedirs(tmp)
            plt.imsave(os.path.join(tmp, imgs), img)
            cnt += 1

    print(f'images counts: {cnt}')


def crop_eval_imgs(crop):
    print('=' * 19 + f' {crop} crop eval imgs ' + '=' * 20)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'	
    test_info = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    path = '/opt/ml/input/data/eval/images'
    new_path = '/opt/ml/input/data/eval/new_imgs'
    mtcnn = MTCNN(keep_all=True, device=device)
    cnt = 0
    
    if os.path.exists(new_path):
        if crop == 'Stay':
            print('Already exists!')
            return
    else:
        os.mkdir(new_path)
    
    for i in tqdm.tqdm(test_info.values):
        img = cv2.imread(os.path.join(path, i[0]))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        boxes, probs = mtcnn.detect(img)
        
        if not isinstance(boxes, np.ndarray):
            img = img[100:400, 50:350, :]
        
        else:
            xmin = int(boxes[0, 0]) - 15
            ymin = int(boxes[0, 1]) - 15
            xmax = int(boxes[0, 2]) + 15
            ymax = int(boxes[0, 3]) + 15
            
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
            
            img = img[ymin:ymax, xmin:xmax, :]

        plt.imsave(os.path.join(new_path, i[0]), img)
        cnt += 1
    print(f'images counts: {cnt}')
