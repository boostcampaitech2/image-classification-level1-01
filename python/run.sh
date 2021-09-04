pip install -r requirements.txt
python modeltest.py
python train.py --crop New
python train.py -m 1 -anum 58 -l focal_loss -e 4
python train.py -m 2 -e 4
python inference.py
