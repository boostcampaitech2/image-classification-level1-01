import argparse
from typing import NewType
import torch
import torch.nn as nn
import torch.optim as optim
import os, tqdm, copy, random, re
from tensorboardX import SummaryWriter
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import f1_score
from data.crop import *
from model.focal_loss import *
from data.dataset import *
from data.kfold import *
import warnings
import wandb

def make_seed(seed):
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    print(f'* Seed: {seed}')


def normal_weights(data):
    normed_weights=torch.FloatTensor([1-(x/sum(data))for x in data])
    return normed_weights

def train(args, nums, params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    warnings.filterwarnings('ignore')
    
    
    writer = SummaryWriter(log_dir=args.log_dir)
    model_name = ['mask', 'age', 'gender']
    tfms = ['mask_transform', 'age_transform', 'gender_transform']
    model_num = [3 ,3, 2]
    best_loss = 100
    best_f1 = 0.0
    
    if args.model == 0:
        weighted = normal_weights([1, 1, 5]).to(device)
    elif args.model == 1:
        if params['age_test_num'] == 59:
            weighted = normal_weights([1281,1227,192]).to(device)
        else:
            weighted = normal_weights([1281,1142,277]).to(device)
    else:
        weighted = normal_weights([1654,1046]).to(device)
    print(f'* Weight: {weighted}')
    
    if params['loss'] == 'cross_entropy_loss':
        criterion = nn.CrossEntropyLoss(weight=weighted).to(device)
    elif params['loss'] == 'focal_loss':
        criterion = FocalLoss(weight=weighted).to(device)
        
    model = EfficientNet.from_pretrained(
        params['net'], num_classes=model_num[args.model]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3
    )
    
    if params['index'] == 'label':
        train_ids, valid_ids = get_idx_label(seed=args.seed)
    else:
        train_ids, valid_ids = get_idx_people(args.seed)
        
    train_data = magv_dataset(
        data_path=args.data_dir, index=[i for i in range(2700*7)],
        transforms=tfms[args.model], resize=args.resize,
        age_test_num=params['age_test_num'], model=args.model
    )
    
    
    valid_data = magv_dataset(
        data_path=args.data_dir, index=valid_ids[4],
        transforms='valid_transform', resize=args.resize,
        age_test_num=params['age_test_num'], model=args.model
    )
    
    train_sampler = RandomSampler(train_data)
    valid_sampler = RandomSampler(valid_data)

    train_dataloader = DataLoader(
        train_data, batch_size=params['batch_size'],
        shuffle=False, sampler=train_sampler, num_workers=4
    )
    
    valid_dataloader = DataLoader(
        valid_data, batch_size=params['batch_size'],
        shuffle=False, sampler=valid_sampler, num_workers=4
    )
    xx=params['epochs']
    for epoch in range(params['epochs']):
        print('-' * 24 + f' {epoch+1}/{xx} Epochs ' + '-' * 24)
        
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            elif phase == 'valid':
                model.eval()
                
            running_loss, running_corrects = 0.0, 0.0
            num_cnt, f1, f1_scr = 0.0, 0.0, 0.0
            
            if phase == 'train':
                for sample in tqdm.tqdm(train_dataloader):
                    inputs = sample['image'].to(device)
                    label = sample['label'].to(device)
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, label)
                        loss.backward()
                        optimizer.step()
                        
                        num_cnt += params['batch_size']
                        running_loss += loss.item()
                        running_corrects += torch.sum(preds==label.data)
                        f1_scr += f1_score(
                            label.detach().cpu(), preds.detach().cpu(),
                            average='macro'
                        )

            elif phase == 'valid':
                for sample in tqdm.tqdm(train_dataloader):
                    inputs = sample['image'].to(device)
                    label = sample['label'].to(device)
                    optimizer.zero_grad()
                    
                    with torch.no_grad():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, label)
                        
                        num_cnt += params['batch_size']
                        running_loss += loss.item()
                        running_corrects += torch.sum(preds==label.data)
                        f1_scr += f1_score(
                            label.detach().cpu(), preds.detach().cpu(),
                            average='macro'
                        )

            epoch_loss = float(running_loss)
            epoch_acc = float(running_corrects/num_cnt*100)
            f1 = f1_scr/(num_cnt/params['batch_size'])
            
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            writer.add_scalar(f'F1-Score/{phase}', f1, epoch)
            ################################# wnadb logging ##############################################
            wandb.log({f'{phase}_accuracy': epoch_acc,f'{phase}_loss': epoch_loss,f'{phase}_f1': f1})
            ##############################################################################################
            print(
                f'** {phase} loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1_score: {f1:.4f}'
            )
            
            if args.checkpoint == 'loss':
                checking = 1 if epoch_loss < best_loss else 0
            else:
                checking = 1 if f1 > best_f1 else 0
                
            if phase == 'valid' and checking:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
                model_name_temp=args.model
                torch.save(best_model, f'{args.log_dir}best_model_{model_name[model_name_temp]}{nums}.pth')
                print(f'*** best_model_{model_name[model_name_temp]}{nums} save!')
            # print('-' * 60)
            
    return model.state_dict()


############################## trainning 함수를 한번 더 감싸줌 #########################################
def make_args():
    
    wandb.init(name="train")
    params = wandb.config
    ######################## 본래 코드에서 최대한 안고치려고 이렇게 했습니다..#############################
    class Args():
        def __init__(self,m):
            self.seed = 777
            self.crop = "Stay"
            self.model=m
            self.checkpoint="loss"
            self.counts = 1
            self.data_dir = '/opt/ml/input/data/train/images'
            self.log_dir = './log'
            self.resize = 312
    
    # args = parser.parse_args()
    ####################### Model Choice [mask:0, age:1, gender:2]#######################
    model_num=0
    #####################################################################################
    args = Args(model_num)
    
    print('=' * 60)
    print('=' * 26 + ' TRAIN ' + '=' * 27)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'* Device: {device}')
    make_seed(args.seed)
    
    model_name = ['mask', 'age', 'gender']
    print(f'* Model: {args.model}: {model_name[args.model]}')
    if args.model == 0:
        print(f'            Normal / Incorrect / Mask')
    elif args.model == 1:
        print(f'            0 - 29 / 30 - {args.age_test_num} / {args.age_test_num+1} - 100')
    else:
        print(f'            Female / Male')
    
    crop_train_imgs(args.crop)
    crop_eval_imgs(args.crop)
    
    for i in range(args.counts):
        print('=' * 24 + f' {i+1}/{args.counts} Counts ' + '=' * 24)
        train(args, i, params)
    print('=' * 60)


if __name__ == '__main__':
    count = 3
    sweep_config = {
        "name" : "Mask Classification Hyper Param Tunning",
        "method" : "bayes",
        "parameters" : {
            "epochs":{
                "distribution": "int_uniform",
                "min": 2,
                "max": 6
            },
            "age_test_num":{
                "distribution": "int_uniform",
                "min": 57,
                "max": 60
            },
            "net":{
                "values" : ['efficientnet-b3', 'efficientnet-b4']
            },
            "loss":{
                "values" : ['cross_entropy_loss', 'focal_loss']
            },
            "batch_size":{
                "values" : [16,32, 64]
            },
            "learning_rate":{
                "values" : [1e-4, 2e-4, 3e-4]
            },
            "index":{
                "values" : ['label', 'person']
            },
            
        },
        "metric":{
            "name": "valid_f1",
            "goal": "maximize"
        },
    }

    sweep_id = wandb.sweep(sweep_config, 
                       project="Mask Classification")
    # train 함수를 한번더 감싼 이유가 agent에서 function을 부를때 인자를 넘기는 방법을 모르겠더라고요
    # 인자를 넘기고 싶으면 coinfg에서 넘기던가 해야될것 같습니다.
    # 일단 저는 그러지 않고 함수를 한번 더 감싸서 처리했습니다!!!
    # 꼭 실행전에 터미널에서 
    # >>> wandb login
    # 하시면 주소 뜨는데 거기서 token 복붙하면 로그인 됩니다(가입먼저는 당연..ㅎ)
    wandb.agent(sweep_id, function=make_args, count=count)