import torchvision 
import random
import numpy as np 
import torch 
from PIL import Image
from PIL import ImageFilter
import pixmix_utils as utils
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

Simsiam_transform = transforms.Compose([       
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8), 
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 6.5])], p=0.5),
                transforms.ToTensor(),
    ])

## PIXMIX
def augment_input(image, aug_severity=1):
    aug_list = utils.augmentations_all 
    op = np.random.choice(aug_list)
    return op(image.copy(), aug_severity)

def pixmix(orig, mixing_pic, mixing_pic2, preprocess):
    assert isinstance(orig, Image.Image)
    assert isinstance(mixing_pic, Image.Image)
    assert isinstance(mixing_pic2, Image.Image)
    
    k, beta = 4, 4
    mixings = utils.mixings
    tensorize = preprocess['tensorize']
    
    mixed = tensorize(augment_input(orig))

    for _ in range(0,np.random.randint(k)+1):
        if np.random.random() < 0.25:
            aug_image_copy = tensorize(augment_input(orig))
        elif np.random.random() < 0.5 or np.random.random() > 0.25:
        # else:
            if np.random.random() < 0.5:
                aug_image_copy = tensorize(mixing_pic)
            else:
                aug_image_copy = tensorize(mixing_pic2)
        else: # np.random.random() < 0.75 or np.random.random() > 0.5:
            aug_image_copy = Simsiam_transform(orig)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)
        mixed = torch.clip(mixed, 0, 1)
    return mixed

