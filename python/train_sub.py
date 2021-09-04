import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os, tqdm, copy, random, re
#from tensorboardX import SummaryWriter
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data.cutmix import *
from data.dataset import *
from data.kfold import *

from model.focal_loss import *
from model.LabelSmoothingCrossEntropy import *
from model.multi_sample_dropout import *

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


def make_seed(seed):
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    print(f'* Seed: {seed}')
    
def cal_class_weight(class_num):
    class_weight = torch.tensor(np.max(class_num)/class_num).to(device=device,dtype=torch.float)
    
    return class_weight

def train(args, nums, train_idx, valid_idx, ensemblenum):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = ['mask', 'age', 'gender']
    model_num=[3,3,2]
    #tfms 설명
    # 0번째 index: train과 valid 공통으로 쓸 경우
    # 1번째 index: train에만 쓸 경우
    # 2번째 index: valid에만 쓸 경우
    # 3번째 index: cutmix에 쓸 경우(우선, cutmix 시, train과 valid 공통
    tfms = ['transform', 'train_transform', 'valid_transform', 'cutmix_transform']
    
    best_model_state=None
    best_f1=0
    best_loss=1000
    early_stop_count=0
        
    #############(mask, age, gender별) class_weight 주기!##############3
    if args.model == 0:
        class_weight = cal_class_weight([13500, 2700, 2700])
    elif args.model == 1:
        if args.age_test_num==59:
            class_weight = cal_class_weight([8967, 8589, 1344])
        else:
            class_weight = cal_class_weight([8967, 7994, 1939])
    else:
        class_weight = cal_class_weight([7294, 11606])
        
    print(f'class_weight: {class_weight}')
        
            
        
        
    ###############ensemble하면 efficientnet_b3 먼저 그 다음 vit_base_patch16_224 이렇게 썼어서 그것을 default로 했습니다.###################
    ########### 어떤 사전학습모델을 쓸지 저장하는 변수 model_n #################
    model_n = args.net
    if args.ensemble==1:
        if ensemblenum==0:
            model_n="efficientnet_b3"
        elif ensemblenum==1:
            model_n="vit_base_patch16_224"
    
    ################# argument로 정한 loss 사용하기 ######################3
    if args.loss =='cross_entropy_loss':
        criterion = nn.CrossEntropyLoss(weight=class_weight).to(device=device)
    elif args.loss =='focal_loss':
        criterion = FocalLoss(weight=class_weight).to(device=device)
    elif args.loss =='Label_smoothing_cross_entropy':
        #label smoothing 시, weight 안 받는다.
        criterion = LabelSmoothingCrossEntropy()
            
    model = timm.create_model(model_n,pretrained=True,num_classes=model_num[args.model]).to(device=device)
        
        
    #################### multi sample dropout 결정 #######################
    if args.multi_sample_dropout==1:
        if "efficientnet" in model_n:
            model.classifier = MultiSampleDropout(model_name=model_n).to(device=device)
        elif "vit" in model_n:
            model.head = MultiSampleDropout(model_name=model_n).to(device=device)
        
    ############### layer learning rate 적용 #######################
    if "efficientnet" in model_n:
        classifier = [p for p in model.classifier.parameters()]
        feature_extractor = [m for n,m in model.named_parameters() if "classifier" not in n]
        params = [{"params": feature_extractor, "lr": args.learning_rate * 0.5},{"params":classifier,"lr": args.learning_rate}]
        
    elif "vit" in model_n:
        classifier = [p for p in model.head.parameters()]
        feature_extractor = [m for n,m in model.named_parameters() if "head" not in n]
        params = [{"params": feature_extractor, "lr": args.learning_rate * 0.5},{"params":classifier,"lr": args.learning_rate}]
        
    ############## optimizer 설정 ########################
    optimizer = optim.Adam(params,lr=args.learning_rate)
        
    ################ scheduler 설정 ########################
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
        
    ############## train transforms 이름 정하기 #########################
    train_transforms=''
    # cutmix할 경우 cutmix만의 하나의 transform을 이용!
    if args.cutmix==1:
        train_transforms = model_n+"_"+model_name[args.model]+"_"+tfms[3]
    
    elif args.model==1:
        train_transforms = model_n+"_"+model_name[args.model]+"_"+tfms[1]
    
    else:
        train_transforms = model_n+"_"+model_name[args.model]+"_"+tfms[0]
    print(train_transforms)    
    
    #################### train dataset 만들기 #############################
    train_data = magv_dataset(args.data_dir, index=train_idx, transforms=train_transforms, resize=args.resize, age_test_num=args.age_test_num, model=args.model,sub=1)
    
    ############## valid transforms 이름 정하기 #########################
    valid_transforms=''
    if args.cutmix==1:
        valid_transforms = model_n+"_"+model_name[args.model]+"_"+tfms[3]
    elif args.model==1:
        valid_transforms = model_n+"_"+model_name[args.model]+"_"+tfms[2]
    else:
        valid_transforms = model_n+"_"+model_name[args.model]+"_"+tfms[0]
    print(valid_transforms)
    
    ############# valid dataset 만들기 #########################
    valid_data = magv_dataset(args.data_dir, index=valid_idx, transforms=valid_transforms, resize=args.resize, age_test_num=args.age_test_num, model=args.model,sub=1)
        
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    ############ cutmix할 시, 변수 설정 ########################
    if args.cutmix==1:
        USE_CUTMIX = True # cutmix를 할 것인가
        beta = 1.0
        cutmix_prob=0.5
        cutmix = CutMix(beta=beta, cutmix_prob=cutmix_prob)
        
    ################# 학습 시작! ###############################
    for epoch in range(1,args.epochs+1):
        print('-' * 19 + f' {epoch}/{args.epochs} Epochs ' + '-' * 19)
        
        # train 시작!
        model.train()
        
        running_loss, running_corrects = 0.0, 0.0
        f1=0.0
        f1_scr=0.0
        num_cnt=0.0
        ###### train! ######
        for sample in tqdm.tqdm(train_dataloader):
            inputs, label = sample['image'].to(device=device,dtype=torch.float), sample['label'].to(device)
            
            optimizer.zero_grad()
            ############## cutmix할 경우!###########
            if args.cutmix==1:
                ratio = np.random.rand(1)
                if USE_CUTMIX:
                    if beta>0 and ratio < cutmix_prob:
                        new_inputs = cutmix.forward(inputs, label)
                        outputs = model(new_inputs['image'])
                        loss = criterion(outputs, new_inputs['label_1']) * new_inputs['lam'] + criterion(outputs,new_inputs['label_2']) * (1. - new_inputs['lam'])
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, label)
            
            ############### cutmix 아닐 경우 #####################
            else:
                outputs = model(inputs)
                loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            num_cnt += args.batch_size
            running_loss += loss.item()
            running_corrects+=torch.sum(preds==label.data)
            f1_scr += f1_score(label.detach().cpu(), preds.detach().cpu(),
                               average='macro')
        
        epoch_loss = float(running_loss)
        epoch_acc = float(running_corrects/num_cnt*100)
        f1 = f1_scr/(num_cnt/args.batch_size)
        
        print( f'train loss: {epoch_loss:.4f} | train Acc: {epoch_acc:.4f} | train F1_score: {f1:.4f}')
        
        #################### valid! ###########################
        running_loss, running_corrects = 0.0, 0.0
        f1=0.0
        f1_scr=0.0
        num_cnt=0.0
        #for sample in valid_dataloader:
        for sample in tqdm.tqdm(valid_dataloader):
            inputs = sample['image'].to(device)
            label = sample['label'].to(device)
            
            with torch.no_grad():
                model.eval()
                
                outputs = model(inputs)
                _, preds = torch.max(outputs,1)
                loss = criterion(outputs,label)
                
                num_cnt+=args.batch_size
                running_loss += loss.item()
                running_corrects += torch.sum(preds==label.data)
                f1_scr += f1_score(label.detach().cpu(), preds.detach().cpu(),average='macro')
        
        epoch_loss = float(running_loss)
        epoch_acc = float(running_corrects/num_cnt*100)
        f1 = f1_scr/(num_cnt/args.batch_size)
        
        print( f'valid loss: {epoch_loss:.4f} | valid Acc: {epoch_acc:.4f} | valid F1_score: {f1:.4f}')
        
        if args.checkpoint == 'loss':
            checking = 1 if epoch_loss < best_loss else 0
        else:
            checking = 1 if f1>best_f1 else 0
        
        ###################### best model 저장 ##################################
        if checking:
            best_f1 = f1
            best_loss = epoch_loss
            best_model_state = model.state_dict()
            early_stop_count=0
            
            ############# 앙상블 안 할 경우, 모델 이름 저장 방법 #################
            if args.ensemble==0:
                torch.save(best_model_state, f'{args.log_dir}best_model_state_{model_name[args.model]}{nums}_sub.pth')
                print(f'best_model_state_{model_name[args.model]}{nums} save!')
            
            ############# 앙상블 할 경우, 모델 이름 저장 방법 #################
            elif args.ensemble==1:
                # 처음은 efficientnet_b3
                if ensemblenum==0:
                    torch.save(best_model_state, f'{args.log_dir}best_model_state_efficientnet_b3_{model_name[args.model]}{nums}_sub.pth')
                    print(f'best_model_state_efficientnet_b3_{model_name[args.model]}{nums} save!')
                # 두 번째는 vit
                elif ensemblenum==1:
                    torch.save(best_model_state, f'{args.log_dir}best_model_state_vit_base_patch16_224_{model_name[args.model]}{nums}_sub.pth')
                    print(f'best_model_state_vit_base_patch16_224_{model_name[args.model]}{nums} save!')
            
        
        else:
            early_stop_count+=1
        
        if early_stop_count==args.early_stop:
            print("early stopped." + " "*30)
            break
    
        
    return model.state_dict()    

    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    # Data and model checkpoints directories
    parser.add_argument('-s', '--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('-l', '--loss', type=str, default='focal_loss', help='cross_entropy_loss, focal_loss, Label_smoothing_cross_entropy (default: cross_entropy_loss)  *(noncutmix)Age model: change in focal_loss *(cutmix)Age model: change in Label_smoothing_cross_entropy')
    parser.add_argument('-m', '--model', type=int, default=1, help='mask:0, age:1, gender:2 (default: 0)')
    parser.add_argument('-n', '--net', type=str, default='efficientnet_b3', help='efficientnet_b3, efficientnet_b4, vit_base_patch16_224 (default: efficientnet_b3) *(noncutmix) Age model: change in efficientnet_b3 & vit_base_patch16_224 *(cutmix) Age model: change in efficientnet_b3')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='1e-4, 2e-4, 3e-4 (defualt: 1e-4)')
    parser.add_argument('-r', '--resize', type=int, default=312, help='(defualt: 312)')
    parser.add_argument('-anum', '--age_test_num', type=int, default=58, help='58, 59 (defualt: 58)') # 나의 모델 기준으로는 30~58세까지 1, 59세부터 2로 잡음
    #parser.add_argument('-c', '--crop', type=str, default='Stay', help='Stay, New (defualt: stay)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='(default: 32)')
    #우선 되는지 확인하기 위해 2 -> 원래는 30 epoch
    parser.add_argument('-e', '--epochs', type=int, default=2, help='(default: 30)')
    parser.add_argument('-cp', '--checkpoint', type=str, default='loss', help='loss, f1 (default: loss)')
    parser.add_argument('-ct', '--counts', type=int, default=4, help='(defualt: 4)')
    
    ############################## 추가 argument 부분 ###########################################
    ## cutmix할지 말지 결정
    parser.add_argument('-cm', '--cutmix', type=int, default=0, help='0, 1 (default: 0 (notcutmix)) * 1 : cutmix')
    ## 앙상블 할지 말지 결정
    parser.add_argument('-em', '--ensemble', type=int, default=0, help='0, 1 (default: 0 (notensemble)) * 1 : ensemble')
    ## multi sample dropout 할지 말지 결정 
    parser.add_argument('-msd', '--multi_sample_dropout', type=int, default=0, help='0, 1 (default: 0 (not multi_sample_dropout)) * 1 : multi_sample_dropout')
    ## scheduler 머 쓸지 결정
    parser.add_argument('-sch', '--scheduler', type=str, default="CosineAnnealingLR", help='CosineAnnealingLR, ReduceLROnPlateau (default: CosineAnnealingLR)')
    ## early stop 결정!
    parser.add_argument('-es','--early_stop', type=int, default=3, help='(default: 3)')
    
    
    
    #Container environment
    #우선 작동을 위해 save_dir을 저의 기준에 적용했습니다.
    parser.add_argument('-log', '--log_dir', type=str, default='./log/', help='(default: ./log/)')
    parser.add_argument('-data', '--data_dir', type=str, default='/opt/ml/input/data/train/images', help='(default: /opt/ml/input/data/train/images')
    
    args = parser.parse_args()
    print(args)
    
    assert args.loss in ['cross_entropy_loss', 'focal_loss', 'Label_smoothing_cross_entropy'], f'Wrong Loss: {args.loss}'
    assert args.model in [0, 1, 2], f'Wrong Model: {args.model}'
    assert args.net in ['efficientnet_b3', 'efficientnet_b4', 'vit_base_patch16_224'], f'Wrong Net: {args.net}'                 
    assert args.age_test_num in [58, 59], f'Wrong Net: {args.age_test_num}'
    assert args.cutmix in [0, 1], f'Wrong Cutmix: {args.cutmix}'
    #assert args.crop in ['Stay', 'New'], f'Wrong Crop: {args.crop}'
    assert args.checkpoint in ['loss', 'f1'], f'Wrong Checkpoint: {args.checkpoint}'
    assert args.ensemble in [0,1], f'Wrong Ensemble: {args.ensemble}'
    assert args.multi_sample_dropout in [0,1], f'Wrong multi_sample_dropout: {args.multi_sample_dropout}'
    assert args.scheduler in ["CosineAnnealingLR", "ReduceLROnPlateau"], f'Wrong scheduler: {args.scheduler}'
    
    
    print('=' * 50)
    print('=' * 21 + ' TRAIN ' + '=' * 21)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    make_seed(args.seed)
    
    model_name = ['mask', 'age', 'gender']
    print(f'Net: {args.net}')
    print(f'Model: {args.model}: {model_name[args.model]}')
    print(f'Loss: {args.loss}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Use cumtix: {args.cutmix}')
    print(f'Use ensemble: {args.ensemble}')
    
    train_ids, valid_ids = get_idx_label(seed=args.seed,sub=1)
    # 앙상블 시 몇 번째 모델인지 count
    ensemblenum=0
    
    #앙상블 안 하는 경우
    if args.ensemble==0:
        print(f"{4} fold training starts...")
        for i in range(4):
            print(f"- {i+1} fold - ")
            train(args,i,train_ids[i],valid_ids[i],ensemblenum)
        print('='*50)
    
    #앙상블 하는 경우
    elif args.ensemble==1:
        print(f"{args.counts} fold efficientnet_b3 model training starts...")
        for i in range(args.counts):
            print(f"- {i+1} fold - ")
            train(args,i,train_ids[i],valid_ids[i],ensemblenum)
        print('='*50)
        ensemblenum+=1
        print(f"{args.counts} fold vit_base_patch16_224 model training starts...")
        for i in range(args.counts):
            print(f"- {i+1} fold - ")
            train(args,i,train_ids[i],valid_ids[i],ensemblenum)
        print('='*50)