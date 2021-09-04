import argparse
import torch
import os, tqdm
import timm
import pandas as pd
from torch.utils.data import DataLoader
from data.dataset import dataset_test
from model.multi_sample_dropout import *

def inference(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    submission = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    
    mask_counts = args.counts // 100
    age_counts = args.counts % 100 // 10
    gender_counts = args.counts % 10
    # 4,4,4
    print(f'Counts: mask: {mask_counts}, age: {age_counts}, gender: {gender_counts}')
    
    ################### mask와 gender 모델 갯수만큼 모델 만들기 ##############################
    for i in range(mask_counts):
        globals()[f'model_mask{i}']= timm.create_model('efficientnet_b4', num_classes=3).to(device)
    
    for i in range(gender_counts):
        globals()[f'model_gender{i}']= timm.create_model('efficientnet_b4', num_classes=2).to(device)
    
    mask_gender_transform="efficientnet_b4_mask_transform" #mask와 gender는 transform이 같음
    
    
    ################### age 모델 갯수만큼 모델 만들기 ##############################
    age_transform=[]
    #cutmix를 age에 적용했을 때, efficientnet_b3+multi_sample_dropoutd을 사용!
    if args.cutmix:
        age_transform.append("cutmix_transform")
        for i in range(age_counts):
            model = timm.create_model('efficientnet_b3',pretrained=True,num_classes=3).to(device=device)
            if args.multi_sample_dropout: # multi_sample_dropout을 했을 때! (저는 age 분류 모델에 다 multi_sampe_dropout을 적용했습니다!) -> default
                model.classifier = MultiSampleDropout(model_name='efficientnet_b3').to(device=device)
            globals()[f'model_age{i}']=model
    
    else:
        # age 부분에 앙상블을 했다면
        ################### 앙상블 age 모델은 vit와 efficientnet_b3 2개 모델 만들기 ##############################
        if args.ensemble:
            age_transform.append("efficientnet_b3_age_valid_transform")
            age_transform.append("vit_base_patch16_224_age_valid_transform")
            for i in range(age_counts):
                model = timm.create_model('efficientnet_b3',pretrained=True,num_classes=3).to(device=device)
                if args.multi_sample_dropout:
                    model.classifier = MultiSampleDropout(model_name='efficientnet_b3').to(device=device)
                globals()[f'efficientnet_b3_model_age{i}']=model
                
            for i in range(age_counts):
                model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=3).to(device=device)
                if args.multi_sample_dropout:
                    model.head = MultiSampleDropout(model_name='vit_base_patch16_224').to(device=device)
                globals()[f'vit_base_patch16_224_model_age{i}']=model
            
        # 앙상블을 안 할 경우
        else:
            age_transform.append("efficientnet_b3_age_valid_transform")
            for i in range(age_counts):
                model = timm.create_model('efficientnet_b3',pretrained=True,num_classes=3).to(device=device)
                if args.multi_sample_dropout:
                    model.classifier = MultiSampleDropout(model_name='efficientnet_b3').to(device=device)
                globals()[f'model_age{i}']=model
    
    
    ######################### mask & gender 저장한 모델 가져오기 ##################################
    for i in range(mask_counts):
        # 원래 경로 ./best_model_mask{i}.pth
        eval(f'model_mask{i}').load_state_dict(torch.load(f'{args.save_dir}best_model_state_mask{i}_sub.pth'))
    
    if mask_counts == 1:
        model_mask0.eval()
    else:
        models_mask = list()
        models_mask.extend([eval(f'model_mask{i}') for i in range(mask_counts)])
    
    for i in range(gender_counts):
        # 원래 경로 ./best_model_gender{i}.pth
        eval(f'model_gender{i}').load_state_dict(torch.load(f'{args.save_dir}best_model_state_gender{i}_sub.pth'))
    
    if gender_counts == 1:
        model_gender0.eval()
    else:
        models_gender = list()
        models_gender.extend([eval(f'model_gender{i}') for i in range(gender_counts)])
    
    
    
    ######################### 앙상블 안 한 age 저장한 모델 가져오기 ##################################
    if len(age_transform)==1:
        for i in range(gender_counts):
            # 원래 경로 ./best_model_age{i}.pth
            eval(f'model_age{i}').load_state_dict(torch.load(f'{args.save_dir}best_model_state_age{i}_sub.pth'))
        if age_counts == 1:
            model_age0.eval()
        else:
            models_age = list()
            models_age.extend([eval(f'model_age{i}') for i in range(age_counts)])
    
    ######################### 앙상블한 age 저장한 모델 가져오기 ##################################
    # transform이 2개 있다는 것은 앙상블이라는 뜻!
    elif len(age_transform)==2:
        for i in range(gender_counts):
            # 원래 경로 ./best_model_age{i}.pth
            eval(f'efficientnet_b3_model_age{i}').load_state_dict(torch.load(f'{args.save_dir}best_model_state_efficientnet_b3_age{i}_sub.pth'))
        
        for i in range(gender_counts):
            eval(f'vit_base_patch16_224_model_age{i}').load_state_dict(torch.load(f'{args.save_dir}best_model_state_vit_base_patch16_224_age{i}_sub.pth'))
            # 원래 경로 ./best_model_age{i}.pth #{args.save_dir}
        
        models_age1 = list()
        models_age2 = list()
        models_age1.extend([eval(f'efficientnet_b3_model_age{i}') for i in range(age_counts)])
        models_age2.extend([eval(f'vit_base_patch16_224_model_age{i}') for i in range(age_counts)])
    
    
    
    
    ########################## mask와 gender부터 예측 시작!##############################
    test_mask_gender_dataset = dataset_test(test_data, transforms=mask_gender_transform,sub=1)
    test_mask_gender_dataloader = DataLoader(test_mask_gender_dataset, batch_size=32, shuffle=False,num_workers=4)
    
    
    mask_answer_logits=[]
    gender_answer_logits=[]
    
    print('Calculating gender & mask inference results..')
    if mask_counts==1 and gender_counts==1:
        
        mask_logits = []
        gender_logits = []
        
        model_mask0.eval()
        model_gender0.eval()
        
        with torch.no_grad():
            
            for sample in test_mask_gender_dataloader:
                
                inputs = sample['image'].to(device)
                
                output_mask = model_mask(inputs)
                output_gender = model_gender(inputs)
                mask_logits.extend(output_mask.cpu().tolist())
                gender_logits.extend(output_gender.cpu().tolist())
                
        mask_answer_logits.append(mask_logits)
        gender_answer_logits.append(gender_logits)
    
    else:
        
        for model_mask, model_gender in tqdm.tqdm(zip(models_mask,models_gender)):
            
            mask_logits = []
            gender_logits = []
            
            model_mask.eval()
            model_gender.eval()
        
            with torch.no_grad():
                
                for sample in test_mask_gender_dataloader:
                    
                    inputs = sample['image'].to(device)
                    
                    output_mask = model_mask(inputs)
                    output_gender = model_gender(inputs)
                    mask_logits.extend(output_mask.cpu().tolist())
                    gender_logits.extend(output_gender.cpu().tolist())
                    
                    
            mask_answer_logits.append(mask_logits)
            gender_answer_logits.append(gender_logits)
            
            
    mask_answer_logits = np.mean(mask_answer_logits,axis=0)
    mask_answer_value = np.argmax(mask_answer_logits,axis=-1) # mask 분류한 array
    
    gender_answer_logits = np.mean(gender_answer_logits,axis=0)
    gender_answer_value = np.argmax(gender_answer_logits,axis=-1) # gender 분류한 array
                
    
    ############################ ensemble 아닌 age 예측 부분 #############################################
    print('Calculating age inference results..')
    age_answer_logits=[]
    if(len(age_transform)==1):
        
        test_age_dataset = dataset_test(data=test_data, transforms=age_transform[0],sub=1)
        test_age_dataloader = DataLoader(test_age_dataset, batch_size=32, shuffle=False,num_workers=4)
        
        if age_counts==1:
            
            age_logits = []
            model_age0.eval()
            
            with torch.no_grad():
                
                for sample in test_age_dataloader:
                    
                    inputs = sample['image'].to(device)
                    output_age = model_age0(inputs)
                    age_logits.extend(output_age.cpu().tolist())
                    
            age_answer_logits.append(age_logits)
        
    
        
        else:
            
            for model_age in tqdm.tqdm(models_age):
                
                age_logits = []
                model_age.eval()
                
                with torch.no_grad():
                    
                    for sample in test_age_dataloader:
                        
                        inputs = sample['image'].to(device)
                        output_age = model_age(inputs)
                        age_logits.extend(output_age.cpu().tolist())
                        
                age_answer_logits.append(age_logits)
                
                
        age_answer_logits = np.mean(age_answer_logits,axis=0)
        age_answer_value = np.argmax(age_answer_logits,axis=-1) # 앙상블 아닌 age 분류한 array
    
    ############################ ensemble age 예측 부분 #############################################
    if(len(age_transform)==2):
        
        #efficient 모델
        test_age_dataset = dataset_test(test_data, transforms=age_transform[0],sub=1)
        test_age_dataloader = DataLoader(test_age_dataset, batch_size=32, shuffle=False,num_workers=4)
        
        
        for model in tqdm.tqdm(models_age1):
            age_logits = []
            model.eval()
            
            with torch.no_grad():
                
                
                for sample in test_age_dataloader:
                    inputs = sample['image'].to(device)
                    output_age = model(inputs)
                    age_logits.extend(output_age.cpu().tolist())
        
            age_answer_logits.append(age_logits)
        
        
        # vit 모델
        test_age_dataset = dataset_test(test_data, transforms=age_transform[1],sub=1)
        test_age_dataloader = DataLoader(test_age_dataset, batch_size=32, shuffle=False,num_workers=4)
        
        
        for model in tqdm.tqdm(models_age2):
            age_logits = []
            model.eval()
            
            
            with torch.no_grad():
                
                for sample in test_age_dataloader:
                    inputs = sample['image'].to(device)
                    output_age = model(inputs)
                    age_logits.extend(output_age.cpu().tolist())
                    
            
            age_answer_logits.append(age_logits)
            
            
    age_answer_logits = np.mean(age_answer_logits,axis=0)
    age_answer_value = np.argmax(age_answer_logits,axis=-1) # ensemble 아닌 age 분류한 array

    ################ 다 구한 값 mapping 하기! ########################
    submission['ans']=mask_answer_value*6 + gender_answer_value*3 + age_answer_value
    submission.to_csv(f'{args.save_dir}{args.name}.csv', index=False)
    
    print('Done!')
        
        
            
        
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #우선 작동을 위해 save_dir을 저의 기준에 적용했습니다.
    parser.add_argument('-save', '--save_dir', type=str, default='./log/', help='(default: ./log/)')
    parser.add_argument('-n', '--name', type=str, default='submission', help='(default: submission)')
    # k-fold 4이므로 444 default
    parser.add_argument('-ct', '--counts', type=int, default=444, help='mask, age, gender (default: 444)')
    parser.add_argument('-r', '--resize', type=int, default=312, help='(default: 312)')
    
    ############################## 추가 argument 부분 ###########################################
    ## cutmix인지에 따라 transform이 다르므로 cutmix 인지를 판별해주는 변수 추가
    parser.add_argument('-cm', '--cutmix', type=int, default=0, help='0, 1 (default: 0 (notcutmix)) * 1 : cutmix')
    ## 앙상블 했는지에 따라 k-fold 4로한 모델 하나 더 불러야한다. -> 제일 좋게 나온 모델이 ensemble한 모델이니 default를 1로 했다!
    parser.add_argument('-em', '--ensemble', type=int, default=1, help='0, 1 (default: 1 (ensemble)) * 0 : notensemble')
    ## 보통 age에 multi sample dropout을 하므로 age 평가를 위한 변수 (age 분류 모델 부를 때, 모델에 뒤에 multi sample dropout을 추가하여 불러야 weight 저장된 모델을 부를 수 있다.) 
    parser.add_argument('-msd','--multi_sample_dropout', type=int, default=0, help='0, 1 (default: 0 (not multi_sample_dropout)) * 1 : multi_sample_dropout')
    
    args = parser.parse_args()
    print(args)
    
    print('=' * 50)
    print('=' * 22 + ' TEST ' + '=' * 22)
    inference(args)
    print('=' * 50)