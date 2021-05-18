# Faster tha nTorch vision 
# good for segmetnation , instance segmetnaion 

import cv2 
import albumentations as A
import numpy as np 
from PIL import Image 
from utils import plot_examples

image = cv2.imread("images/cat.jpeg")
image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bboxes = [[13, 170, 224, 410]]
# bbox according to dataset. 


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

    ], bbbox_params  = A.BboxParams(format='pascal_voc', label_fields= [])

)

images_list = [image]
saved_bbox = [bbox[0]]
for i in range(15):
    augumentations = transform(image=image, bboxes= bboxes)
    augument_img = augumentations["image"]
    images_list.append(augument_img)
    saved_bbox.append(augumentations["bboxes"][0])


plot_examples(images_list)