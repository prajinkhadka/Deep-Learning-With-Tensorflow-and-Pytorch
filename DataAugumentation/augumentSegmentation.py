# Faster tha nTorch vision 
# good for segmetnation , instance segmetnaion 

import cv2 
import albumentations as A
import numpy as np 
from PIL import Image 
from utils import plot_examples

image = Image.open("images/elon.jpeg")
mask = Image.open("images/mask.jpeg")

transform = A.Compose(
    [
        A.Resize(width = 1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode= cv2.BORDER_CONSTANT), 
        A.HorizontalFlip(p=0.3),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5)
        ])

    ]
)

images_list = [image]
image = np.array(image)
mask = np.array(mask)

for i in range(15):
    augumentations = transform(image=image, mask=mask)
    augument_img = augumentations["image"]
    augument_mask = augumentations["mask"]

    images_list.append(augument_img)
    images_list.append(augument_mask)


plot_examples(images_list)