import torch
import numpy as np
import random

def make_seed(seed):
	'''
	Set the seed of the random number generator to a fixed value.
	Fixed values for reproducing results.
	'''
	torch.manual_seed(seed) # CPU
	torch.cuda.manual_seed(seed) # GPU
	torch.cuda.manual_seed_all(seed) # multi-GPU
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)

	print(f'* Seed: {seed}')
