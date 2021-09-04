import os, re
import numpy as np
import pandas as pd

def prepare_mask_data(data_path,subject=0):
    # data_path: /opt/ml/input/data/train/images
    data = {'path':[], 'mask':[]}
    
    for labels in os.listdir(data_path):
        label = None
        # labels 중 하나 -> 000001_female_Asian_45
        if labels[0]==".":
            continue
        
        sub = os.path.join(data_path,labels)
        
        for img in os.listdir(sub):
            if subject==0:
                if img[0] == '.': continue
                if img.find('normal') != -1:
                    label = 0
                elif img.find('incorrect') != -1:
                    label = 1
                else:
                    label = 2
            elif subject==1:
                # img 중 하나 -> incorrect_mask.jpg
                if img[0] =='.':
                    continue
                if "incorrect" in img:
                    label=1
                elif "normal" in img:
                    label=2
                elif "mask" in img:
                    label=0
            
            data['path'].append(os.path.join(sub,img))
            data['mask'].append(label)
    
    return pd.DataFrame(data)

#age_num은 몇살까지로 끊을 지 정하는 변수, data는 앞에 마스크와 경로 있는 dataframe
def prepare_age_gender_data(data,age_num):
    
    temp = {'age': [], 'gender': []}
    train = pd.read_csv('/opt/ml/input/data/train/train.csv')
    
    # ~29는 0, 30~age_num: 1, 60~100: 2로 mapping -> columns은 age로 만든다.
    tmp = pd.DataFrame(np.digitize(
        train.iloc[:,3].values, [29, age_num, 100], [0,1,2])
                       ,columns=['age'])
    
    train = train.drop(['race'],axis=1) # 모두 asia로 되어 있어 제거
    train = train.drop(['age'],axis=1) #age를 제거하고 tmp로 대채할 것이다.
    train = pd.concat([train,tmp],axis=1)
    train['gender'] = train['gender'].map({'female':0,'male':1}) # female이면 0, male이면 1로 mapping
    
    # 지금까지 age, gender, mask모두 mapping 수행 완료
    for i in range(len(train)):
        # male => female
        if train.iloc[i, 3] in ['004432_male_Asian_43','001498-1_male_Asian_23']:
            train.iloc[i, 1] = 0
        # female => male
        elif train.iloc[i, 3] in ['006359_female_Asian_18','006360_female_Asian_18',
                                  '006361_female_Asian_18','006362_female_Asian_18',
                                  '006363_female_Asian_18','006364_female_Asian_18']:
            train.iloc[i, 1] = 1
    
    for index, i in enumerate(data.iloc[:,0].values):
        classes = i.split('/')[-2] # 000001_female_Asian_45 이런 부분
        for idx in train.values:
            #idx[2]는 train.csv의 path 부분
            if classes == idx[2]:
                temp['age'].append(idx[3]) #tmp로 mapping한 나이
                temp['gender'].append(idx[1]) #gender 0,1 mapping한 것
    
    return pd.DataFrame(temp) #age와 gender dataframe

# data는 이미지 경로와 mask, labels는 age와 gender
def concat_data_sub(data,labels):
    #url = './input/data/train/images/'
    
    #mask & age & gender
    data = pd.concat([data,labels['age'],labels['gender']],axis=1)
    
        #?
#     for idx, i in enumerate(data.values):
#         tmp = i[0] #이미지 경로
#         tmp = re.sub('images', 'new_imgs', tmp)
#         data.iloc[idx,0]=tmp

    return data

def concat_data(data, labels):
    url = './input/data/train/images/'
    
    data = pd.concat([data, labels['age'], labels['gender']], axis=1)
    
    for i in range(len(data)):
        # normal => incorrect
        if data.iloc[i,0] in [url+'000020_female_Asian_50/normal.jpg',
                              url+'004418_male_Asian_20/normal.jpg',
                              url+'005227_male_Asian_22/normal.jpg']:
            data.iloc[i,1] = 0
        elif data.iloc[i,0] in [url+'000020_female_Asian_50/incorrect_mask.jpg',
                                url+'004418_male_Asian_20/incorrect_mask.jpg',
                                url+'005227_male_Asian_22/incorrect_mask.jpg']:
            data.iloc[i, 1] = 1

    for idx, i in enumerate(data.values):
        tmp = i[0]
        tmp = re.sub('images', 'new_imgs', tmp)
        data.iloc[idx, 0] = tmp
    return data
    


        