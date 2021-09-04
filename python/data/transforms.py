import albumentations as A

class magv_transforms():
    def __init__(self, mode=None, resize=312):
        if mode == 'mask_transform':
            self.transform = A.Compose([
                A.OneOf([
                    A.RandomBrightness(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.RandomContrast(p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.Perspective(p=1.0),
                    A.Rotate(p=0.5, limit=20, border_mode=1)
                ], p=0.5),
                A.Compose([
                    A.Resize(resize, resize),
                    A.Normalize()
                ])
            ])
        elif mode == 'age_transform':
            self.transform = A.Compose([
                A.OneOf([
                    A.RandomGridShuffle(grid=(2, 2), p=1.0),
                    A.Perspective(p=1.0)
                ], p=0.5),
                A.GaussNoise(p=0.5),
                A.Rotate(limit=20, p=0.5, border_mode=1),
                A.Compose([
                    A.Resize(resize, resize),
                    A.Normalize()
                ])
            ])
        elif mode == 'gender_transform':
            self.transform = A.Compose([
                A.OneOf([
                    A.Perspective(p=1.0)
                ], p=0.5),
                A.GaussNoise(p=0.5),
                A.Rotate(limit=20, p=0.5, border_mode=1),
                A.Compose([
                    A.Resize(resize, resize),
                    A.Normalize()
                ])
            ])
        elif mode == 'valid_transform':
            self.transform = A.Compose([
                A.Resize(resize, resize),
                A.Normalize()
            ])

    def __call__(self, image):
        return self.transform(image=image)['image']