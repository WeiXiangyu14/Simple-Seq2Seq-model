import os
import sys
import re
import string
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, RepeatVector


DATA_PATH = os.path.join(os.getcwd(),'data')
STRUCT_PATH = os.path.join(os.getcwd(),'struct')
MODEL_STRUCT_FILE = 'model_struct.json'
MODEL_WEIGHTS_FILE = 'model_weights.h5'
TRAIN_FILE = 'train.txt'
TEST_FILE = 'test.txt'
BEGIN_SYMBOL = '^'
END_SYMBOL = '$'
CHAR_SET = set(string.ascii_lowercase + BEGIN_SYMBOL + END_SYMBOL)
CHAR_NUM = len(CHAR_SET)
CHAR_TO_INDICES = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25, '^':26, '$':27}
INDICES_TO_CHAR = {i:c for c, i in CHAR_TO_INDICES.items()}
MAX_INPUT_LEN = 14
MAX_OUTPUT_LEN = 16
HIDDEN_SIZE = 1024
NON_ALPHA_PAT = re.compile('[^a-z]')


def is_vowel(char):
    return char in ('a', 'e', 'i', 'o', 'u')


def transform(word):
    if word[-3:] == "man":
        return word[:-3] + "men"
    if word[-2:] == "fe":
        return word[:-2] + "ves"
    if word[-1] == "f":
        return word[:-1] + "ves"
    if word[-1] == "y" and not is_vowel(word[-2]):
        return word[:-1] + "ies"
    if word[-1] == "s" or word[-1] == "x" or word[-1] == "o":
        return word + "es"
    if word[-2:] == "sh" or word[-2:] == "ch":
        return word + "es"
    return word + "s"


def vectorize(word, seq_len, vec_size):
    # use one-hot method to vectorize a word
    vec = np.zeros((seq_len, vec_size), dtype=int)
    for i, ch in enumerate(word):
        vec[i, CHAR_TO_INDICES[ch]] = 1
    for i in range(len(word), seq_len):
        vec[i, CHAR_TO_INDICES[END_SYMBOL]] = 1
    return vec


def build_data():
    words_file = os.path.join(DATA_PATH, TRAIN_FILE)
    # words = []
    # for w in open(words_file, 'r').readlines():
    #     words.append(w.replace("\n", ""))
    words = [
        w.lower().strip() for w in open(words_file, 'r').readlines()
        if w.strip() != '' and not NON_ALPHA_PAT.findall(w.lower().strip())
    ]

    plain_x = []
    plain_y = []
    for w in words:
        plain_x.append(BEGIN_SYMBOL + w)
        plain_y.append(BEGIN_SYMBOL + transform(w))

    train_x = np.zeros((len(words), MAX_INPUT_LEN, CHAR_NUM), dtype=int)
    train_y = np.zeros((len(words), MAX_OUTPUT_LEN, CHAR_NUM), dtype=int)
    for i in range(len(words)):
        train_x[i] = vectorize(plain_x[i], MAX_INPUT_LEN, CHAR_NUM)
        train_y[i] = vectorize(plain_y[i], MAX_OUTPUT_LEN, CHAR_NUM)

    return train_x, train_y


def build_model(input_size, seq_len, hidden_size):
    model = Sequential()
    model.add(LSTM(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(RepeatVector(seq_len))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=input_size, activation="linear")))
    model.compile(loss="mse", optimizer='adam')

    return model


def get_predict_result(model, word):
    x = np.zeros((1, MAX_INPUT_LEN, CHAR_NUM), dtype=int)
    word = BEGIN_SYMBOL + word.lower().strip() + END_SYMBOL
    x[0] = vectorize(word, MAX_INPUT_LEN, CHAR_NUM)

    pred = model.predict(x)[0]
    return(''.join([
        INDICES_TO_CHAR[i] for i in pred.argmax(axis=1)
        if INDICES_TO_CHAR[i] not in (BEGIN_SYMBOL, END_SYMBOL)
    ]))


def train(epoch):
    x, y = build_data()
    model = build_model(CHAR_NUM, MAX_OUTPUT_LEN, HIDDEN_SIZE)
    model.fit(x, y, validation_split=0.1, batch_size=128, nb_epoch=epoch)
    open(os.path.join(STRUCT_PATH, MODEL_STRUCT_FILE), 'w').write(model.to_json())
    model.save_weights(os.path.join(STRUCT_PATH, MODEL_WEIGHTS_FILE), overwrite=True)


def test(word):
    model = model_from_json(open(os.path.join(STRUCT_PATH, MODEL_STRUCT_FILE), 'r').read())
    model.compile(loss="mse", optimizer='adam')
    model.load_weights(os.path.join(STRUCT_PATH, MODEL_WEIGHTS_FILE))

    if word == "":
        testnum = 0
        rightnum = 0
        for w in open(os.path.join(DATA_PATH, TEST_FILE)).readlines():
            testnum += 1
            w = w.replace("\n", "")
            print(get_predict_result(model, w) + " " + transform(w))

            if get_predict_result(model, w) == transform(w):
                rightnum += 1

        print("Accuracy: %f" %(rightnum / testnum))
    else:
        print("\nPredict result:")
        print(get_predict_result(model, word))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        if len(sys.argv) > 2:
            train(int(sys.argv[2]))
        else:
            train(300)
    elif sys.argv[1] == 'test':
        if len(sys.argv) > 2:
            test(sys.argv[2])
        else:
            test("")
    else:
        print("wrong parameters")
