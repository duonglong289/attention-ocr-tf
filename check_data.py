
import os 
import numpy as np 
import cv2
from attentionocr.image import ImageUtil

image_util = ImageUtil(32, 320)
phase = "train"

if phase == "train":
    with open('./dataset/src-train.txt', 'r') as f:
        images = f.readlines()
    with open('./dataset/tgt-train.txt', 'r') as f:
        labels = f.readlines()
else:
    with open('./dataset/src-val.txt', 'r') as f:
        images = f.readlines()
    with open('./dataset/tgt-val.txt', 'r') as f:
        labels = f.readlines()

for i, path_img in enumerate(images):
    image_path = os.path.join("./dataset/images", path_img.strip())
    image = cv2.imread(image_path)
    # if augment:
    #     image = seq.augment_image(image)
    
    if any([e==0 for e in image.shape]):
        print("Image None: ", image_path)
        continue
    try:
        image = image_util.preprocess(image)
    except:
        print(image_path)
        continue
        
    # image = np.expand_dims(image, axis=2)
    label = labels[i].strip()
    label = label.split(" ")
    text = ""
    print(i)
    # for char in label:
    #     if char == "\;":
    #         char == " "
    #     text += char