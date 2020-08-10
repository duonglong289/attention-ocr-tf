import string

import numpy as np


# default_vocabulary = list(string.ascii_lowercase) + list(string.digits) + [' ', '-', '.', ':', '?', '!', '<', '>', '#', '@', '(', ')', '$', '%', '&']
# default_vocabulary = list(string.ascii_uppercase) + list(string.digits) + [' ']
default_vocabulary = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']

class Vocabulary:
    pad = '<PAD>'
    sos = '<SOS>'
    eos = '<EOS>'
    unk = '<UNK>'

    def __init__(self, vocabulary: list = default_vocabulary):
        self._characters = [self.pad, self.sos, self.eos, self.unk] + sorted(vocabulary)
        self._character_index = dict([(char, i) for i, char in enumerate(self._characters)])
        self._character_reverse_index = dict((i, char) for char, i in self._character_index.items())

    def is_special_character(self, char_idx):
        return self._characters[char_idx] in [self.pad, self.sos, self.eos]

    def one_hot_encode(self, txt: str, length: int, sos: bool = False, eos: bool = True) -> np.ndarray:
        txt = txt.upper()
        txt = list(txt)
        txt = txt[:length - int(sos) - int(eos)]
        txt = [c if c in self._characters else self.unk for c in txt]
        if sos:
            txt = [self.sos] + txt
        if eos:
            txt = txt + [self.eos]
        txt += [self.pad] * (length - len(txt))
        encoding = np.zeros((length, len(self)), dtype='float32')
        for char_pos, char in enumerate(txt):
            if char in self._character_index:
                encoding[char_pos, self._character_index[char]] = 1.
            else:
                encoding[char_pos, self._character_index[self.unk]] = 1.

            
        return encoding

    def one_hot_decode(self, one_hot: np.ndarray, max_length: int) -> str:
        text = ''
        for sample_index in np.argmax(one_hot, axis=-1):
            sample = self._character_reverse_index[sample_index]
            if sample == self.eos or sample == self.pad or len(text) > max_length:
                break
            text += sample
        return text

    def __len__(self):
        return len(self._characters)


if __name__ == "__main__":
    import numpy as np 
    voc = Vocabulary()
    print(default_vocabulary)

    # import ipdb; ipdb.set_trace()
    # a = voc.one_hot_encode("", 1, sos=True, eos=False)
    # print(len(voc._character_reverse_index))
    # print(voc._character_reverse_index)
 