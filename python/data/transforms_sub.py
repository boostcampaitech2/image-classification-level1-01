import albumentations as A

class magv_transforms_sub():
    def __init__(self,mode=None,resize=None):
        if "efficientnet_b4" in mode or "gender" in mode or "mask" in mode:
            if "vit" in mode:
                resize=224
                self.transform=A.Compose([
                    A.CenterCrop(380,380),
                    A.Resize(224,224),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            else:
                self.transform=A.Compose([
                    A.CenterCrop(380,380),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            
        
        elif mode == "efficientnet_b3_age_train_transform":
            self.transform = A.Compose([
                A.CenterCrop(350,300),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        
        elif mode == "efficientnet_b3_age_valid_transform":
            self.transform = A.Compose([
                A.CenterCrop(350,300),
                A.Normalize()
            ])
        
        elif mode == "vit_base_patch16_224_age_valid_transform":
            resize=224
            self.transform = A.Compose([
                A.CenterCrop(350,300),
                A.Resize(resize,resize), #224,224
                A.Normalize()
            ])
        
        elif mode == "vit_base_patch16_224_age_train_transform":
            resize=224
            self.transform= A.Compose([
                A.CenterCrop(350,300),
                A.Resize(resize,resize), #224,224
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize()
            ])
        
        
        elif "cutmix_transform" in mode:
            if "vit" in mode:
                resize=224
                self.transform = A.Compose([
                    A.CenterCrop(350,350),
                    A.Resize(resize,resize), #224,224
                    A.Normalize()
                ])
            else:
                resize=312
                self.transform = A.Compose([
                    A.CenterCrop(350,350),
                    A.Resize(resize,resize), #312,312
                    A.Normalize()
                ])
            
            
    def __call__(self,image):
        return self.transform(image=image)['image']