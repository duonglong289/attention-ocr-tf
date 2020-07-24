import argparse

import tensorflow as tf
from attentionocr import Vectorizer, AttentionOCR, synthetic_data_generator, Vocabulary, CSVDataSource


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--epoch_size', type=int, default=8, required=False)
    parser.add_argument('--image_width', type=int, default=320, required=False)
    parser.add_argument('--max_txt_length', type=int, default=42, required=False)
    parser.add_argument('--batch_size', type=int, default=8, required=False)
    parser.add_argument('--validate_every_steps', type=int, default=10, required=False)
    parser.add_argument('--data_directory', type=str, default=None, required=False)
    parser.add_argument('--pretrained_model', type=str, default=None, required=False)
    parser.add_argument('--model_name', type=str, default='trained.h5', required=False)

    args = parser.parse_args()

    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=args.image_width, max_txt_length=args.max_txt_length)
    model = AttentionOCR(vocabulary=voc, max_txt_length=args.max_txt_length)

    model.load("trained.h5")
    import ipdb; ipdb.set_trace()
    # print(model)