# Image Classification

## Tree

```python
.
├── data
│   ├── crop.py
│   ├── dataset.py
│   ├── kfold.py
│   ├── prepare_data.py
│   └── transforms.py
├── etc
│   └── randomness.py
├── log
├── model
│   └── focal_loss.py
├── evaluation.py
├── inference.py
└── train.py
```

## Reproduce the final result

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

## Common Options

### train.py

|  short  |       long        | description                                  | default                         |
| :-----: | :---------------: | -------------------------------------------- | ------------------------------- |
|  `-s`   |     `--seed`      | random seed                                  | 777                             |
|  `-c`   |     `--crop`      | Stay(If you have crop images) / New          | Stay                            |
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

---

## SUB)

