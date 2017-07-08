import os
import random
import re


DATA_PATH = os.path.join(os.getcwd(),'data')
TRAIN_FILE = 'train.txt'
TEST_FILE = 'test.txt'
WORD_FILE = 'words.txt'

wordfile = open(os.path.join(DATA_PATH, WORD_FILE), 'r')
testfile = open(os.path.join(DATA_PATH, TEST_FILE), 'w')
trainfile = open(os.path.join(DATA_PATH, TRAIN_FILE), 'w')
wordlist = wordfile.read().split()
print(len(wordlist))
maxlen = 0
for w in wordlist:
    w = re.findall('[a-z]', w)
    w = ''.join(w)
    if len(w) > 12 or len(w) < 3:
        continue
    if len(w) > maxlen:
        maxlen = len(w)
    if random.random() > 0.8:
        testfile.write(w + '\n')
    else:
        trainfile.write(w + '\n')
print(maxlen)
wordfile.close()
testfile.close()
trainfile.close()