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
img = cv2.imread("/media/geneous/01D62877FB2A4900/Techainer/OCR/test_model/attention-ocr-tf/data_test_idcard/MicrosoftTeams-image.png")
input_data = image_util.preprocess(img)
input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
print(input_data.shape)
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
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
#           'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']
vocabulary = list(string.ascii_uppercase) + list(string.digits) + [' ']

_characters = [pad, sos, eos, unk] + sorted(vocabulary)

_character_index = dict([(char, i) for i, char in enumerate(_characters)])

_character_reverse_index = dict((i, char) for char, i in _character_index.items())
# {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: ' ', 5: '0', 6: '1', 7: '2', 8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 
# 13: '8', 14: '9', 15: 'A', 16: 'B', 17: 'C', 18: 'D', 19: 'E', 20: 'F', 21: 'G', 22: 'H', 23: 'I', 24: 'J', 25: 'K', 26: 'L', 
# 27: 'M', 28: 'N', 29: 'O', 30: 'P', 31: 'Q', 32: 'R', 33: 'S', 34: 'T', 35: 'U', 36: 'V', 37: 'W', 38: 'X', 39: 'Y', 40: 'Z'}}


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

# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
# print("output_data", output_data)
y_pred = np.squeeze(output_data, axis=0)  # squeeze the batch index out
text = one_hot_decode(y_pred, 20)
print(text)