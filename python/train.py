import argparse
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
    normed_weights=torch.FloatTensor([1-(x/sum(data))for x in data]).to(device)
    return normed_weights

def train(args, nums):
    warnings.filterwarnings('ignore')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter(log_dir=args.log_dir)
    model_name = ['mask', 'age', 'gender']
    tfms = ['mask_transform', 'age_transform', 'gender_transform']
    model_num = [3 ,3, 2]
    best_loss = 100
    best_f1 = 0.0
    
    if args.model == 0:
        weighted = normal_weights([1, 1, 5])
    elif args.model == 1:
        if args.age_test_num == 59:
            weighted = normal_weights([1281,1227,192])
        else:
            weighted = normal_weights([1281,1142,277])
    else:
        weighted = normal_weights([1654,1046])
    print(f'* Weight: {weighted}')
    
    if args.loss == 'cross_entropy_loss':
        criterion = nn.CrossEntropyLoss(weight=weighted).to(device)
    elif args.loss == 'focal_loss':
        criterion = FocalLoss(weight=weighted).to(device)
        
    model = EfficientNet.from_pretrained(
        args.net, num_classes=model_num[args.model]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3
    )
    
    if args.index == 'label':
        train_ids, valid_ids = get_idx_label(seed=args.seed)
    else:
        train_ids, valid_ids = get_idx_people(args.seed)
        
    train_data = magv_dataset(
        data_path=args.data_dir, index=[i for i in range(2700*7)],
        transforms=tfms[args.model], resize=args.resize,
        age_test_num=args.age_test_num, model=args.model
    )
    
    
    valid_data = magv_dataset(
        data_path=args.data_dir, index=valid_ids[4],
        transforms='valid_transform', resize=args.resize,
        age_test_num=args.age_test_num, model=args.model
    )
    
    train_sampler = RandomSampler(train_data)
    valid_sampler = RandomSampler(valid_data)

    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=False, sampler=train_sampler, num_workers=4
    )
    
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=False, sampler=valid_sampler, num_workers=4
    )

    for epoch in range(args.epochs):
        print('-' * 24 + f' {epoch+1}/{args.epochs} Epochs ' + '-' * 24)
        
        
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
                        
                        num_cnt += args.batch_size
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
                        
                        num_cnt += args.batch_size
                        running_loss += loss.item()
                        running_corrects += torch.sum(preds==label.data)
                        f1_scr += f1_score(
                            label.detach().cpu(), preds.detach().cpu(),
                            average='macro'
                        )

            epoch_loss = float(running_loss)
            epoch_acc = float(running_corrects/num_cnt*100)
            f1 = f1_scr/(num_cnt/args.batch_size)
            
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            writer.add_scalar(f'F1-Score/{phase}', f1, epoch)
            
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
                torch.save(best_model, f'{args.log_dir}best_model_{model_name[args.model]}{nums}.pth')
                print(f'*** best_model_{model_name[args.model]}{nums} save!')
            # print('-' * 60)
            
    return model.state_dict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('-s', '--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('-c', '--crop', type=str, default='New', help='Stay, New (defualt: Stay)')
    parser.add_argument('-m', '--model', type=int, default=0, help='mask:0, age:1, gender:2 (default: 0)')
    parser.add_argument('-anum', '--age_test_num', type=int, default=59, help='58, 59 (defualt: 59)')
    parser.add_argument('-r', '--resize', type=int, default=312, help='(defualt: 312)')
    parser.add_argument('-n', '--net', type=str, default='efficientnet-b3', help='efficientnet-b3, efficientnet-b4 (defualt: efficientnet-b3)')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy_loss', help='cross_entropy_loss, focal_loss (default: cross_entropy_loss)  *Age modle: change in focal_loss')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='(defualt: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='(defualt: 2)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='1e-4, 2e-4, 3e-4 (defualt: 1e-4)')
    parser.add_argument('-i', '--index', type=str, default='label', help='label, person (defualt: label)')
    parser.add_argument('-cp', '--checkpoint', type=str, default='loss', help='loss, f1 (defualt: loss)')
    parser.add_argument('-ct', '--counts', type=int, default=5, help='(defualt: 5)')
    
    # Container environment
    parser.add_argument('-data', '--data_dir', type=str, default='/opt/ml/input/data/train/images', help='(default: /opt/ml/input/data/train/images)')
    parser.add_argument('-log', '--log_dir', type=str, default='./log/', help='(default: ./log/)')
    
    args = parser.parse_args()
    
    assert args.loss in ['cross_entropy_loss', 'focal_loss'], f'Wrong Loss: {args.loss}'
    assert args.model in [0, 1, 2], f'Wrong Model: {args.model}'
    assert args.net in ['efficientnet-b3', 'efficientnet-b4'], f'Wrong Net: {args.net}'
    assert args.age_test_num in [58, 59], f'Wrong Age test num: {args.net}'
    assert args.crop in ['Stay', 'New'], f'Wrong Crop: {args.crop}'
    assert args.checkpoint in ['loss', 'f1'], f'Wrong Checkpoint: {args.checkpoint}'
    assert args.index in ['label', 'person'], f'Wrong Checkpoint: {args.index}'

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
    print(f'* Resize: {args.resize}')
    print(f'* Net: {args.net}')
    print(f'* Loss: {args.loss}')
    print(f'* KFold: {args.index}')
    print(f'* Batch_size: {args.batch_size}')
    print(f'* Epoch: {args.epochs}')
    print(f'* Learning_rate: {args.learning_rate}')
    print(f'* Checkpoint: {args.checkpoint}')
    print(f'* Counts: {args.counts}')
    print(f'* Data: {args.data_dir}')
    print(f'* Log: {args.log_dir}')
    
    crop_train_imgs(args.crop)
    crop_eval_imgs(args.crop)
    
    for i in range(args.counts):
        print('=' * 24 + f' {i+1}/{args.counts} Counts ' + '=' * 24)
        train(args, i)
    print('=' * 60)
