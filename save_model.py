import argparse
import tensorflow as tf 
from attentionocr import Vectorizer, AttentionOCR, Vocabulary
from attentionocr.image import ImageUtil
import cv2
import numpy as np
import glob
import os 


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 32
BATCH_SIZE = 1
CHANNEL = 3
MAX_LENGTH = 20

class AttentionOCRModule(tf.Module):
    def __init__(self, model):
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL), dtype=tf.float32)])
    def input_img(self, input_img):
        # decoder_spec = tf.zeros([BATCH_SIZE, 1, len_vocab], dtype=tf.float32)
        decoder_spec = np.zeros((BATCH_SIZE, 1, len_vocab), dtype=np.float32)
        decoder_spec[0][0][1] = 1.0
        decoder_spec = tf.convert_to_tensor(decoder_spec)
        results = self.model([input_img, decoder_spec])
        return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--len_vocab', type=int, default=41, required=False)
    parser.add_argument('--image_width', type=int, default=320, required=False)
    parser.add_argument('--image_height', type=int, default=32, required=False)
    parser.add_argument('--max_txt_length', type=int, default=20, required=False)
    parser.add_argument('--pretrained_model', type=str, default=None, required=False)
    parser.add_argument('--model_pb_dir', type=str, default='pb_model', required=False)
    parser.add_argument('--model_tflite_path', type=str, default='saved_model.tflite', required=False)

    args = parser.parse_args()
    global len_vocab
    len_vocab = args.len_vocab

    #MAX_LENGTH
    #global IMAGE_HEIGHT
    # global BATCH_SIZE
    #global IMAGE_WIDTH

    saved_pb_dir = args.model_pb_dir
    saved_tflite_path = args.model_tflite_path
    
    # Model init
    voc = Vocabulary()
    vec = Vectorizer(vocabulary=voc, image_width=args.image_width, max_txt_length=20)
    model_att = AttentionOCR(vocabulary=voc, max_txt_length=MAX_LENGTH)
    # model = model.build_inference_model()
    
    # load 
    if args.pretrained_model is not None:
        model_att.load_weights(args.pretrained_model)
    
    model = model_att.build_inference_model()
    module = AttentionOCRModule(model)
    tf.saved_model.save(module, saved_pb_dir, signatures={"input_img": module.input_img})

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_pb_dir, signature_keys=["input_img"])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    open(args.model_tflite_path, "wb").write(tflite_model)

    
   









