import math
import numpy as np
import pickle

with open('input.txt', 'r') as f:
    input_file = f.read()

chars = sorted(list(set(input_file)))
print(chars)

char2idx = { c: i for (i, c) in enumerate(chars) }
print(char2idx)
idx2char = { i: c for (i, c) in enumerate(chars)}
print(idx2char)

def encode_data(data):
    return [char2idx[char] for char in data]

def decode_data(indices):
    return [idx2char[idx] for idx in indices]

test_start = math.floor(0.90*len(input_file))
train_data = np.array(encode_data(input_file[:test_start]), dtype=np.uint16)
test_data = np.array(encode_data(input_file[test_start:]), dtype=np.uint16)

print(train_data)
print(train_data.shape)
train_data.tofile('train.bin')
test_data.tofile('test.bin')

meta = {
    'itc': idx2char,
    'cti': char2idx,
    'size': len(char2idx),
}

with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
print(meta)
