import numpy as np
import tensorflow as tf
import cv2
import string 
from attentionocr.image import ImageUtil
image_util = ImageUtil(32, 320)

interpreter = tf.lite.Interpreter(model_path="saved_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors information.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
print(input_shape)

# Cac kí tự đặc biệt
pad = '<PAD>'
# Start of sentence
sos = '<SOS>'
# End of sentence
eos = '<EOS>'
# Unknown character
unk = '<UNK>'

# Vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
#           'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '-', '.', '#']
vocabulary = list(string.ascii_uppercase) + list(string.digits) + [' ', '-', '.', '#']
# string.ascii_lowercase

_characters = [pad, sos, eos, unk] + sorted(vocabulary)
# _characters = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', ' ', '#', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 
#                   'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

_character_index = dict([(char, i) for i, char in enumerate(_characters)])
# _character_index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, ' ': 4, '#': 5, '-': 6, '.': 7, '0': 8, '1': 9, '2': 10, '3': 11, '4': 12, '5': 13, 
# '6': 14, '7': 15, '8': 16, '9': 17, 'A': 18, 'B': 19, 'C': 20, 'D': 21, 'E': 22, 'F': 23, 'G': 24, 'H': 25, 'I': 26, 'J': 27, 'K': 28, 
# 'L': 29, 'M': 30, 'N': 31, 'O': 32, 'P': 33, 'Q': 34, 'R': 35, 'S': 36, 'T': 37, 'U': 38, 'V': 39, 'W': 40, 'X': 41, 'Y': 42, 'Z': 43}

_character_reverse_index = dict((i, char) for char, i in _character_index.items())
# _character_reverse_index = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: ' ', 5: '#', 6: '-', 7: '.', 8: '0', 9: '1', 10: '2', 11: '3', 12: '4', 13: '5', 14: '6', 
# 15: '7', 16: '8', 17: '9', 18: 'A', 19: 'B', 20: 'C', 21: 'D', 22: 'E', 23: 'F', 24: 'G', 25: 'H', 26: 'I', 27: 'J', 28: 'K', 29: 'L', 30: 'M', 31: 'N', 32: 'O', 33: 'P',
# 34: 'Q', 35: 'R', 36: 'S', 37: 'T', 38: 'U', 39: 'V', 40: 'W', 41: 'X', 42: 'Y', 43: 'Z'}


def one_hot_decode( one_hot: np.ndarray, max_length: int) -> str:
    text = ''
    for sample_index in np.argmax(one_hot, axis=-1):
        sample = _character_reverse_index[sample_index]
        if sample == eos or sample == pad or len(text) > max_length:
            break
        text += sample
    return text


interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
# import ipdb; ipdb.set_trace()
# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("output_data", output_data)
y_pred = np.squeeze(output_data, axis=0)  # squeeze the batch index out
text = one_hot_decode(y_pred, 20)
