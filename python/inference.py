import argparse
import torch
import os, tqdm
import pandas as pd
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from data.dataset import dataset_test


def inference(args):
	classes = {
		'201': 0, '211': 1, '221': 2,
		'200': 3, '210': 4, '220': 5,
		'101': 6, '111': 7, '121': 8,
		'100': 9, '110': 10, '120': 11,
		'001': 12, '011': 13, '021': 14,
		'000': 15, '010': 16, '020': 17
	}

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	test_data = pd.read_csv('/opt/ml/input/data/eval/info.csv')
	submission = pd.read_csv('/opt/ml/input/data/eval/info.csv')
	
	mask_counts = args.counts // 100
	age_counts = args.counts % 100 // 10
	gender_counts = args.counts % 10

	print(f'* Counts: mask: {mask_counts}')
	print(f'          age: {age_counts}')
	print(f'          gender: {gender_counts}')

	for i in range(mask_counts):
		globals()[f'model_mask{i}'] = EfficientNet.from_name(
				'efficientnet-b3', num_classes=3
				).to(device)
	for i in range(age_counts):
		globals()[f'model_age{i}'] = EfficientNet.from_name(
				'efficientnet-b3', num_classes=3
				).to(device)
	for i in range(gender_counts):
		globals()[f'model_gender{i}'] = EfficientNet.from_name(
				'efficientnet-b3', num_classes=2
				).to(device)
	
	for i in range(mask_counts):
		eval(f'model_mask{i}').load_state_dict(torch.load(f'./log/best_model_mask{i}.pth'))
	for i in range(age_counts):
		eval(f'model_age{i}').load_state_dict(torch.load(f'./log/best_model_age{i}.pth'))
	for i in range(gender_counts):
		eval(f'model_gender{i}').load_state_dict(torch.load(f'./log/best_model_gender{i}.pth'))

	if mask_counts == 1:
		model_mask0.eval()
	else:
		models_mask = list()
		models_mask.extend([eval(f'model_mask{i}') for i in range(mask_counts)])
	if age_counts == 1:
		model_age0.eval()
	else:
		models_age = list()
		models_age.extend([eval(f'model_age{i}') for i in range(age_counts)])
	if gender_counts == 1:
		model_gender0.eval()
	else:
		models_gender = list()
		models_gender.extend([eval(f'model_gender{i}') for i in range(gender_counts)])
	
	test_dataset = dataset_test(
			test_data, transforms='valid_transform'
			)
	test_dataloader = DataLoader(
			test_dataset, batch_size=32, shuffle=False
			)
	total_result = list()

	print('=' * 14 + ' Calculating inference results.. ' + '=' * 14)
	for sample in tqdm.tqdm(test_dataloader):
		with torch.no_grad():
			inputs = sample['image'].to(device)

			if mask_counts == 1:
				output_mask = model_mask0(inputs)
			else:
				output_mask = 0
				for model in models_mask:
					model.eval()
					output_mask += model(inputs)
			
			if age_counts == 1:
				output_age = model_age0(inputs)
			else:
				output_age = 0
				for model in models_age:
					model.eval()
					output_age += model(inputs)

			if gender_counts == 1:
				output_gender = model_gender0(inputs)
			else:
				output_gender = 0
				for model in models_gender:
					model.eval()
					output_gender += model(inputs)

			_, preds_mask = torch.max(output_mask, 1)
			_, preds_age = torch.max(output_age, 1)
			_, preds_gender = torch.max(output_gender, 1)

			for mask, age, gender in zip(
					preds_mask, preds_age, preds_gender
					):
				ans = list()
				ans.append(mask.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				ans.append(age.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				ans.append(gender.detach().cpu().numpy().astype('|S1').tobytes().decode('utf-8'))
				dt = ''.join(ans)

				total_result.append(classes[dt])

	submission['ans'] = total_result
	submission.to_csv(
			f'{args.save_dir}{args.name}.csv', index=False
			)
	print('*** Done!')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-n', '--name', type=str, default='submission', help='(default: submission)')
	parser.add_argument('-ct', '--counts', type=int, default=555, help='mask, age, gender (default: 555)')
	parser.add_argument('-s', '--save_dir', type=str, default='./log/', help='(default: ./log/)')

	args = parser.parse_args()
	
	print('=' * 60)
	print('=' * 25 + ' INFERENCE ' + '=' * 24)
	print(f'* Name: {args.name}')
	print(f'* Save: {args.save_dir}')
	
	inference(args)
	print('=' * 60)


