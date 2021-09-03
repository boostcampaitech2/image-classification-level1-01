import os, re
import numpy as np
import pandas as pd

def prepare_mask_data(data_path):
	'''
	Create a new table with labeled path and mask.
	'''
	data = {'path': [], 'mask': []}

	for labels in os.listdir(data_path):
		label = None

		if labels[0] == '.': continue
		sub = os.path.join(data_path, labels)

		for img in os.listdir(sub):
			if img[0] == '.': continue
			if img.find('normal') != -1:
				label = 0
			elif img.find('incorrect') != -1:
				label = 1
			else:
				label = 2
			data['path'].append(os.path.join(sub, img))
			data['mask'].append(label)

	return pd.DataFrame(data)


def prepare_age_gender_data(data, age_num):
	'''
	Create a new table with labeled age and gender.
	'''
	temp = {'age': [], 'gender': []}
	train = pd.read_csv('/opt/ml/input/data/train/train.csv')

	tmp = pd.DataFrame(np.digitize(
				train.iloc[:, 3].values, [29, age_num, 100], [0, 1, 2]
				), columns=['age'])

	train = train.drop(['race'], axis=1)
	train = train.drop(['age'], axis=1)
	train = pd.concat([train, tmp], axis=1)
	train['gender'] = train['gender'].map({'female': 0, 'male': 1})

	for i in range(len(train)):
		# male => female
		if train.iloc[i, 3] in ['004432_male_Asian_43','001498-1_male_Asian_23']:
			train.iloc[i, 1] = 0
		# female => male
		elif train.iloc[i, 3] in ['006359_female_Asian_18','006360_female_Asian_18',
							'006361_female_Asian_18','006362_female_Asian_18',
							'006363_female_Asian_18','006364_female_Asian_18']:
			train.iloc[i, 1] = 1

	for index, i in enumerate(data.iloc[:, 0].values):
		classes = i.split('/')[-2]
		for idx in train.values:
			if classes == idx[2]:
				temp['age'].append(idx[3])
				temp['gender'].append(idx[1])

	return pd.DataFrame(temp)


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
