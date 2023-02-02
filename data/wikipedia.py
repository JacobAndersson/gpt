import tiktoken
import datasets
import tiktoken
import numpy as np
import os

#num tokens 50257

tokenizer = tiktoken.get_encoding('gpt2')

def encode(row):
    tokens = tokenizer.encode_ordinary(row['text'])
    tokens.append(tokenizer.eot_token)

    return {'tokens': tokens, 'len': len(tokens) }

data = datasets.load_dataset('wikitext', 'wikitext-2-v1')

train_data = data['train'].filter(lambda x: len(x['text']) > 0)
test_data = data['test'].filter(lambda x: len(x['text']) > 0)
print(train_data)

processsed_train = train_data.map(encode, remove_columns=['text'])
processed_test = test_data.map(encode, remove_columns=['text'])

def save(data, split):
    data_length = sum(map(lambda x: x['len'], data))
    pth = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    array_file = np.memmap(pth, dtype=np.uint16, mode="w+", shape=(data_length,))
    
    counter = 0
    for row in data:
        tokens = row['tokens']
        array_file[counter:counter+row['len']] = tokens
        counter += row['len']
    array_file.flush

save(processsed_train, 'train')
save(processed_test, 'test')
