import string
from glob import glob
from random import randint, choice
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from imgaug import augmenters as iaa

from .image import ImageUtil
from .vectorizer import Vectorizer
from .vocabulary import default_vocabulary
# from image import ImageUtil
# from vectorizer import Vectorizer
# from vocabulary import default_vocabulary, Vocabulary
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.append(".")




# seq = iaa.SomeOf((0, 2), [
#     iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
#     iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
#     iaa.Invert(1.0),
#     iaa.MotionBlur(k=10)
# ])

seq = iaa.Sequential([
    iaa.Sometimes(0.2, iaa.OneOf(
        [
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
        ]
    )),

    iaa.Sometimes(0.5, iaa.OneOf(
        [
            iaa.Multiply((0.5, 1.5)),
            iaa.MotionBlur(k=(3, 7)),
            iaa.GaussianBlur((1, 3)),
            iaa.MedianBlur(k=(1, 5)),
            iaa.AverageBlur(k=(2, 4)),
            iaa.BilateralBlur(d=(1, 3), sigma_color=250, sigma_space=250)
        ]
    )),
    iaa.Sometimes(0.3, iaa.SomeOf(2,
        [
            iaa.Add((-20, 20), per_channel=0.5),
            iaa.LinearContrast((0.5, 1.5)),
            iaa.GammaContrast(gamma=(0.45, 1.15)),
            iaa.AddToHueAndSaturation((-20, 20))
        ]
    )),
    iaa.Sometimes(0.5, iaa.OneOf([
        iaa.OneOf(
            [
                iaa.Affine(
                    scale=(0.9, 1.05),
                    rotate=(-5, 5),
                    translate_percent=(-0.05, 0.05),
                    cval=255,
                    mode='constant',
                    fit_output=True),
                iaa.Affine(
                    scale=(0.9, 1.05),
                    shear=(-5, 5),
                    translate_percent=(-0.05, 0.05),
                    cval=255,
                    mode='constant',
                    fit_output=True)
            ]
        ),
        iaa.OneOf(
            [
                iaa.ElasticTransformation(alpha=(0.2, 1.3),
                                        sigma=0.4),
                iaa.PiecewiseAffine(scale=(0.001, 0.009)),
                iaa.PerspectiveTransform(scale=(0, 0.02), keep_size=False),
            ])
        ]
    )),
    iaa.Sometimes(0.1, iaa.Grayscale()),
])



def random_font():
    fontname = choice(list(glob('./synthetic/fonts/*.ttf', recursive=True)))
    font = ImageFont.truetype(fontname, size=randint(24, 32))
    return font


def rand_pad():
    return randint(5, 35), randint(5, 35), randint(0, 3), randint(10, 13)


def random_string(length: Optional[int] = None):
    if length is None:
        length = randint(4, 20)

    if randint(0, 1) == 0:
        random_file = choice(list(glob('./synthetic/texts/*.txt')))
        with open(random_file, 'r') as f:
            random_txt = f.readlines()
        random_txt = choice(random_txt)
        end = len(random_txt) - length
        if end > 0:
            start = randint(0, end)
            random_txt = random_txt[start:start+length].strip()
            if len(random_txt) > 1:
                return random_txt

    letters = list(string.ascii_uppercase) + default_vocabulary
    return (''.join(choice(letters) for _ in range(length))).strip()


def random_background(height, width):
    background_image = choice(list(glob('./synthetic/images/*.jpg')))
    original = Image.open(background_image)
    L = original.convert('L')
    original = Image.merge('RGB', (L, L, L))
    left = randint(0, original.size[0] - height)
    top = randint(0, original.size[1] - width)
    right = left + height
    bottom = top + width
    return original.crop((left, top, right, bottom))


def generate_image(text: str, augment: bool) -> Tuple[np.array, str]:
    font = random_font()
    txt_width, txt_height = font.getsize(text)
    left_pad, right_pad, top_pad, bottom_pad = rand_pad()
    height = left_pad + txt_width + right_pad
    width = top_pad + txt_height + bottom_pad
    image = random_background(height, width)

    stroke_sat = int(np.array(image).mean())
    sat = int((stroke_sat + 127) % 255)
    mask = Image.new('L', (height, width))
    canvas = ImageDraw.Draw(mask)
    canvas.text((left_pad, top_pad), text, fill=sat, font=font, stroke_fill=stroke_sat, stroke_width=2)
    lower = int(-10 + (txt_width / 32))
    upper = int(10 - (txt_width / 32))
    if upper < lower:
        upper = lower
    mask = mask.rotate(randint(lower, upper))
    image.paste(mask, (0, 0), mask)

    image = np.array(image)

    if augment:
        image = seq.augment_image(image)
    image = image_util.preprocess(image)
    # print("hehe", image.shape)
    return image, text.lower()


def synthetic_data_generator(vectorizer: Vectorizer, epoch_size: int = 1000, augment: bool = False, is_training: bool = False):

    def synthesize():
        for _ in range(epoch_size):
            image, text = generate_image(random_string(), augment)
            decoder_input, decoder_output = vectorizer.transform_text(text, is_training)
            yield image, decoder_input, decoder_output

    return synthesize


def load_data_onmt(data_dir, phase, vectorizer: Vectorizer, image_width=224, image_height=32, augment: bool = False, is_training: bool = False):
    image_util = ImageUtil(image_height=image_height, image_width=image_width)

    def synthesize():
        if phase == "train":
            with open(os.path.join(data_dir, 'src-train.txt'), 'r') as f:
                images = f.readlines()
            with open(os.path.join(data_dir, 'tgt-train.txt'), 'r') as f:
                labels = f.readlines()
        else:
            with open(os.path.join(data_dir, 'src-val.txt'), 'r') as f:
                images = f.readlines()
            with open(os.path.join(data_dir, 'tgt-val.txt'), 'r') as f:
                labels = f.readlines()

        for i, path_img in enumerate(images):
            image_path = os.path.join(data_dir, "images", path_img.strip())
            image = cv2.imread(image_path)
            if augment:
                image = seq.augment_image(image)

            try:
                image = image_util.preprocess(image)
            except:
                continue
            # image = np.expand_dims(image, axis=2)
            label = labels[i].strip()
            label = label.split(" ")
            text = ""
            for char in label:
                if char == "\\;":
                    char = " "
                text += char
            decoder_input, decoder_output = vectorizer.transform_text(text, is_training)

            yield image, decoder_input, decoder_output

    return synthesize


if __name__=="__main__":
    import tensorflow as tf
    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=320, max_txt_length=20)
    train_set = load_data_onmt("dataset", "val",vec, augment=False, is_training=True)
    subset = train_set()
    for i, (img, decode_in, decode_out, text ) in enumerate(subset):
        indices = tf.argmax(decode_out,axis=1)
        indices = indices.numpy()

        print(indices)
        print(text)
        # break
        # plt.imshow(img)
        # plt.show()
        
        if i == 10:
            break
