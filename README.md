# 부스트캠프 AI테크 2기 P-Stage Level1-01 팀 코드

# Image Classification

## :one: Abstract

### 1. 목표

- COVID-19 Pandemic 상황 속 마스크 착용 유무 판단 시스템 구축

- 마스크 착용 여부 , 성별 , 나이 총 세가지 Column이 존재하며 총 18개의 Class로 구분하는 모델 

### 2. 결과

#### Best Model

- Data Augmentation

    >  ALL: Perspective, Rotate(limit=20), Resize(312, 312), Normalize
    >
    > Mask: RandomBrightness, HueSaturationValue, RandomContrast
    >
    > Age: RandomGridShuffle, GaussNoise
    >
    > Gender: GaussNoise

- 5_fold, label별

- Model : EfficientNet B3

   > Mask : CrossEntropy Loss / 64 / 1e-4 / 2 * 5
   >
   > Age: FocalLoss / 64 / 1e-4 / 4 * 5
   >
   > Gender : CrossEntropy Loss / 64/1e-4 / 4 * 5

- Optimizer : Adam

- Scheduler : ReduceLROnPlateau(mode='min', factor=0.2, patience=3)

- Input Image Size : (312, 312)

#### Result : 80% Accuracy , 0.761 F1 Score / 전체 8등

### 3. 팀 구성

* 정유석(팀장) : 모델 개발
* 박상하(팀원) : 논문 리뷰 및 자료 조사
* 박진형(팀원) : 모델 개발 및 Wandb
* 백운경(팀원) : 자료 조사 및 모델 학습
* 이상은(팀원) : 모델 개발
* 이홍규(팀원) : 모델 개발 및 다양한 실험 진행


## :two: Tree

```python
.
├── data
│   ├── crop.py
│   ├── dataset.py
│   ├── kfold.py
│   ├── prepare_data.py
│   ├── cutmix.py
│   ├── transforms_sub.py
│   └── transforms.py
├── log
├── model
│   ├── multi_sample_dropout.py
│   ├── LabelSmoothingCrossEntropy.py
│   └── focal_loss.py
├── evaluation.py
├── inference.py
├── train.py
├── modeltest.py
├── inference_sub.py
├── train_sub.py
├── hyperParamTune.py
└── run.sh
```

## :three: Reproduce the final result

### Total run

   `sh run.sh`

### Training

- Mask

   `python train.py`

- Age

   `python train.py -m 1 -anum 58 -l focal_loss -e 4` 

- Gender

   `python train.py -m 2 -e 4` 

### Inference

`python inference.py`

### Evaluation

`python evaluation.py -anum 58`

## :four: Reproduce sub results

### 1. Ensemble

#### Training

- Mask

  `python train_sub.py -m 0 -l cross_entropy_loss -sch CosineAnnealingLR -n efficientnet_b4 -e 30 -cm 0 -em 0 -msd 0`

- Age

  `python train_sub.py -m 1 -l focal_loss -sch CosineAnnealingLR -n efficientnet_b3 -e 30 -cm 0 -em 1 -msd 1` 

- Gender

  `python train_sub.py -m 2 -l cross_entropy_loss -sch CosineAnnealingLR -n efficientnet_b4 -e 30 -cm 0 -em 0 -msd 0`

#### Inference

   `python inference_sub.py -cm 0 -em 1 -msd 1`

### 2. Cutmix

#### Training

- Mask

  `python train_sub.py -m 0 -l cross_entropy_loss -sch CosineAnnealingLR -n efficientnet_b4 -e 30 -cm 0 -em 0 -msd 0`

- Age

  `python train_sub.py -m 1 -l Label_smoothing_cross_entropy -sch ReduceLROnPlateau -n efficientnet_b3 -e 30 -cm 1 -em 0 -msd 1` 

- Gender

  `python train_sub.py -m 2 -l cross_entropy_loss -sch CosineAnnealingLR -n efficientnet_b4 -e 30 -cm 0 -em 0 -msd 0`

#### Inference

   `python inference_sub.py -cm 1 -em 0 -msd 1`

### 3. UnitTest

   `python modeltest.py`

### 4. Hyper Parameter Tunning(wandb)

   `wandb login`

   `python hyperParamTune.py`
      
## :five: Options

### train.py

|  short  |       long        | description                                  | default                         |
| :-----: | :---------------: | -------------------------------------------- | ------------------------------- |
|  `-s`   |     `--seed`      | random seed                                  | 777                             |
|  `-c`   |     `--crop`      | Stay(If you have crop images) / New          | New                            |
|  `-m`   |     `--modle`     | mask: 0 / age: 1 / gender: 2                 | 0                               |
| `-anum` | `--age_test_num`  | age classes: 0 - 29, - (58 / 59), -100       | 59                              |
|  `-r`   |    `--resize`     | image input size                             | 312                             |
|  `-n`   |      `--net`      | efficientnet-b3 / efficientnet-b4            | efficientnet-b3                 |
|  `-l`   |      `-loss`      | cross_entropy_loss / focal_loss              | cross_entropy_loss              |
|  `-bs`  |  `--batch_size`   | batch_size                                   | 64                              |
|  `-e`   |    `--epochs`     | epoch                                        | 2                               |
|  `-lr`  | `--learning_rate` | learning rate                                | 1e-4 (0.0001)                   |
|  `-i`   |     `--index`     | 5-kfold, Split indexes by (label / person)   | label                           |
|  `-cp`  |  `--checkpoint`   | update model parameter with best (loss / f1) | loss                            |
|  `-ct`  |    `--counts`     | repeat counts of model                       | 5                               |
| `-data` |   `--data_dir`    | .                                            | /opt/ml/input/data/train/images |
| `-log`  |    `--log dir`    | .                                            | ./log/                          |


### inference.py

| short |     long     | description                                                  | default    |
| :---: | :----------: | ------------------------------------------------------------ | ---------- |
| `-n`  |   `--name`   | name of csv                                                  | submission |
| `-ct` |  `--counts`  | repeat counts of model (example: 123 => mask 1, age 2, gender 3) | 555        |
| `-s`  | `--save_dir` | .                                                            | ./log/     |

### evaluation.py

|  short  |       long       | description                                                  | default                         |
| :-----: | :--------------: | ------------------------------------------------------------ | ------------------------------- |
|  `-ct`  |    `--counts`    | repeat counts of model (example: 123 => mask 1, age 2, gender 3) | 555                             |
| `-anum` | `--age_test_num` | age classes: 0 - 29, - (58 / 59), -100                       | 59                              |
| `-data` |   `--data_dir`   | .                                                            | /opt/ml/input/data/train/images |
|  `-s`   |   `--save_dir`   | .                                                            | ./log/                          |

### train_sub.py

|  short  |           long           | description                                                  | default                         |
| :-----: | :----------------------: | ------------------------------------------------------------ | ------------------------------- |
|  `-s`   |         `--seed`         | random seed                                                  | 777                             |
|  `-m`   |        `--modle`         | mask: 0 / age: 1 / gender: 2                                 | 0                               |
| `-anum` |     `--age_test_num`     | age classes: 0 - 29, - (58 / 59), -100                       | 58                              |
|  `-r`   |        `--resize`        | image input size                                             | 312                             |
|  `-n`   |         `--net`          | efficientnet_b3 / efficientnet_b4 / vit_base_patch16_224     | efficientnet_b3                 |
|  `-l`   |         `-loss`          | cross_entropy_loss / focal_loss / Label_smoothing_cross_entropy | cross_entropy_loss              |
|  `-bs`  |      `--batch_size`      | batch_size                                                   | 32                              |
|  `-e`   |        `--epochs`        | epoch                                                        | 2                               |
|  `-lr`  |    `--learning_rate`     | learning rate                                                | 1e-4 (0.0001)                   |
|  `-cp`  |      `--checkpoint`      | update model parameter with best (loss / f1)                 | loss                            |
|  `-ct`  |        `--counts`        | repeat counts of model                                       | 5                               |
| `-data` |       `--data_dir`       | .                                                            | /opt/ml/input/data/train/images |
| `-log`  |       `--log dir`        | .                                                            | ./log/                          |
|  `-cm`  |        `--cutmix`        | decide whether to cutmix or not                              | 0                               |
|  `-em`  |       `--ensemble`       | decide whether to ensemble or not                            | 0                               |
| `-msd`  | `--multi_sample_dropout` | decide whether to multi sample dropout or not                | 0                               |
| `-sch`  |      `--scheduler`       | CosineAnnealingLR / ReduceLROnPlateau                        | CosineAnnealingLR               |
|  `-es`  |      `--early_stop`      | decide early stop                                            | 3                               |

### inference_sub.py

| short  |           long           | description                                                  | default    |
| :----: | :----------------------: | ------------------------------------------------------------ | ---------- |
|  `-n`  |         `--name`         | name of csv                                                  | submission |
| `-ct`  |        `--counts`        | repeat counts of model (example: 123 => mask 1, age 2, gender 3) | 444        |
|  `-s`  |       `--save_dir`       | .                                                            | ./log/     |
|  `-r`  |        `--resize`        | image size                                                   | 312        |
| `-cm`  |        `--cutmix`        | decide whether to cutmix or not                              | 0          |
| `-em`  |       `--ensemble`       | decide whether to ensemble or not                            | 1          |
| `-msd` | `--multi_sample_dropout` | decide whether to multi sample dropout or not                | 0          |

