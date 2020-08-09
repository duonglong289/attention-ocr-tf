import argparse

import tensorflow as tf
from attentionocr import Vectorizer, AttentionOCR, synthetic_data_generator, Vocabulary, CSVDataSource

from attentionocr.image import ImageUtil
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--image_width', type=int, default=320, required=False)
    parser.add_argument('--max_txt_length', type=int, default=20, required=False)
    parser.add_argument('--batch_size', type=int, default=1, required=False)
    parser.add_argument('--validate_every_steps', type=int, default=10, required=False)
    parser.add_argument('--data_directory', type=str, default=None, required=False)
    parser.add_argument('--pretrained_model', type=str, default=None, required=False)
    parser.add_argument('--model_name', type=str, default='trained.h5', required=False)

    args = parser.parse_args()

    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=args.image_width, max_txt_length=args.max_txt_length)
    model = AttentionOCR(vocabulary=voc, max_txt_length=args.max_txt_length)
    model.load_weights("snapshots/snapshot-20.h5")
    image_util = ImageUtil(32,320)
    
    images = glob.glob("data_test_idcard/*")
    # import ipdb; ipdb.set_trace()
    for path in images:
        image = cv2.imread(path)
        image = image_util.preprocess(image)
        # import matplotlib.pyplot as plt
        # text = model.predict([image])
        model.visualise([image])
        # print(text)
        # plt.imshow(image)
        # plt.show()
        # print(image.shape)