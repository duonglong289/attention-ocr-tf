import argparse

import tensorflow as tf
from attentionocr import Vectorizer, AttentionOCR, synthetic_data_generator, Vocabulary, CSVDataSource, load_data_onmt

genos = 9999

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--image_height', type=int, default=64, required=False)
    parser.add_argument('--image_width', type=int, default=448, required=False)
    parser.add_argument('--max_txt_length', type=int, default=20, required=False)
    parser.add_argument('--batch_size', type=int, default=128, required=False)
    parser.add_argument('--validate_every_steps', type=int, default=20, required=False)
    parser.add_argument('--data_directory', type=str, default="dataset", required=False)
    parser.add_argument('--pretrained_model', type=str, default=None, required=False)
    parser.add_argument('--model_name', type=str, default='trained.h5', required=False)

    args = parser.parse_args()

    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_height=args.image_height ,image_width=args.image_width, max_txt_length=args.max_txt_length)
    model = AttentionOCR(vocabulary=voc, max_txt_length=args.max_txt_length)

    # Load data
    train_data = load_data_onmt(args.data_directory, "train", vec, augment=False, is_training=True)
    validation_data = load_data_onmt(args.data_directory, "val", vec, augment=False)

    train_gen = tf.data.Dataset.from_generator(train_data, output_types=(tf.float32, tf.float32, tf.float32))
    validation_gen = tf.data.Dataset.from_generator(validation_data, output_types=(tf.float32, tf.float32, tf.float32))

    # Load pretrained model
    if args.pretrained_model is not None:
        pretrained_model_path = args.pretrained_model 
        if pretrained_model_path.endwiths(".h5"):
            model.load_weights(pretrained_model_path)
        else:
            model.load(args.pretrained_model)
    
    # Start training
    model.fit_generator(train_gen, epochs=args.epochs, batch_size=args.batch_size, validation_data=validation_gen, validate_every_steps=args.validate_every_steps)
    if args.model_name:
        model.save(args.model_name)

