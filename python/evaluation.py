import argparse
import torch
import os, tqdm
import pandas as pd
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from data.dataset import dataset_valid
from data.prepare_data import *


def evaluation(args):
	classes = {
		'201': 0, '211': 1, '221': 2,
		'200': 3, '210': 4, '220': 5,
		'101': 6, '111': 7, '121': 8,
		'100': 9, '110': 10, '120': 11,
		'001': 12, '011': 13, '021': 14,
		'000': 15, '010': 16, '020': 17
	}

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	mask_counts = args.counts // 100
	age_counts = args.counts % 100 // 10
	gender_counts = args.counts % 10
	
	print(f'* Counts: mask: {mask_counts}')
	print(f'          age: {age_counts}')
	print(f'          gender: {gender_counts}')
	
	f1 = 0
	f1_sc = 0

	data_mask = prepare_mask_data('/opt/ml/input/data/train/images')
	data_age_gender = prepare_age_gender_data(data_mask, args.age_test_num)
	data = concat_data(data_mask, data_age_gender)
	data_tmp = data.iloc[int(len(data) * 0.8):, :]

	model_mask = EfficientNet.from_name('efficientnet-b3', num_classes=3).to(device)
	model_age = EfficientNet.from_name('efficientnet-b3', num_classes=3).to(device)
	model_gender = EfficientNet.from_name('efficientnet-b3', num_classes=2).to(device)
	
	model_mask.load_state_dict(torch.load(f'{args.save_dir}best_model_mask0.pth'))
	model_age.load_state_dict(torch.load(f'{args.save_dir}best_model_age0.pth'))
	model_gender.load_state_dict(torch.load(f'{args.save_dir}best_model_gender0.pth'))
	
	model_mask.eval()
	model_age.eval()
	model_gender.eval()
	
	valid_test_data = dataset_valid(data_tmp, transforms='valid_transform')
	valid_dataloader = DataLoader(
			valid_test_data, batch_size=16, shuffle=False
			)

	print('=' * 25 +' F1-Score ' + '=' * 25)
	with torch.no_grad():
		for sample in tqdm.tqdm(valid_dataloader):
			inputs = sample['image'].to(device)
	
			output_mask = model_mask(inputs)
			output_age = model_age(inputs)
			output_gender = model_gender(inputs)
	
			_, preds_mask = torch.max(output_mask, 1)
			_, preds_age = torch.max(output_age, 1)
			_, preds_gender = torch.max(output_gender, 1)
	
			for mask, age, gender, label_mask, label_age, label_gender in zip(
					preds_mask, preds_age, preds_gender,
					sample['label_mask'], sample['label_age'], sample['label_gender']
					):
				ans = list()
				label = list()
				
				ans.append(mask.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				ans.append(age.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				ans.append(gender.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				dt = ''.join(ans)
				
				label.append(label_mask.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				label.append(label_age.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				label.append(label_gender.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				lb = ''.join(label)
	
				ans = np.zeros((18, ))
				pred = np.zeros((18, ))
				
				ans[classes[lb]] = 1
				pred[classes[dt]] = 1
				
				f1_sc += f1_score(ans, pred, average='macro')

	print(f'*** F1-Score: {f1_sc/(len(data) * 0.2)}')
    
    
if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-ct', '--counts', type=int, default=555, help='mask, age, gender (default: 555)')
	parser.add_argument('-anum', '--age_test_num', type=int, default=59, help='58, 59 (default: 59)')
	parser.add_argument('-data', '--data_dir', type=str, default='/opt/ml/input/data/train/images', help='(default: /opt/ml/input/data/train/images)')
	parser.add_argument('-s', '--save_dir', type=str, default='./log/', help='(default: ./log/)')

	args = parser.parse_args()

	assert args.age_test_num in [58, 59], f'Wrong Age test num: {args.anum}'

	print('=' * 60)
	print('=' * 24 + ' EVAALUATION ' + '=' * 23)
	print(f'* Age : 0 - 29 / 30 - {args.age_test_num} / {args.age_test_num+1} - 100')
	print(f'* Data: {args.data_dir}')
	print(f'* Save: {args.save_dir}')
	
	evaluation(args)
	print('=' * 60)


