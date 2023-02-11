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

#data = datasets.load_dataset('wikitext', 'wikitext-2-v1')
data = datasets.load_dataset('openwebtext', cache_dir='/mnt/storage/openwebtext')

if 'test' in data:
    train_data = data['train']
    test_data = data['test']
else: 
    split_dataset = data["train"].train_test_split(test_size=0.0005, shuffle=True)
    train_data = split_dataset['train']
    test_data = split_dataset['test']

train_data = train_data
test_data = test_data

processsed_train = train_data.map(encode, remove_columns=['text'], num_proc=4)
processed_test = test_data.map(encode, remove_columns=['text'], num_proc=4)

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
