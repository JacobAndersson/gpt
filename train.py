import numpy as np
import torch
import torch.nn as nn
import model
import pickle
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
block_size = 64
num_epochs = 16
n_layers = 10
embedding_dim = 768
internal_dim = 4*embedding_dim
n_heads = 8

learning_rate = 6e-5
min_learning_rate = 6e-6
warmup_iterations = 2000
decay_iterations = 60000
max_iterations = 60000
testing_interval = 1000

with open('meta.pkl', 'rb') as f:
    pkl = pickle.load(f)
    vocab_size = pkl['size']
    char2idx = pkl['cti']
    idx2char = pkl['itc']

train_idx = np.memmap('train.bin', dtype=np.uint16)
val_idx = np.memmap('test.bin', dtype=np.uint16)

def get_batch(split='train'):
    data = train_idx if split == 'train' else val_idx
    rand = torch.randint(low=0, high=len(train_idx) - block_size, size=(batch_size,))

    x = torch.stack([torch.from_numpy((train_idx[x: x+block_size]).astype(np.int64)) for x in rand])
    y = torch.stack([torch.from_numpy((train_idx[x+1: x+block_size+1]).astype(np.int64)) for x in rand])

    x = x.to(device)
    y = y.to(device)

    return x,y


model = model.GPT(vocab_size=vocab_size, embedding_dim=embedding_dim, block_size=block_size, n_layers=n_layers, internal_dim=internal_dim, n_heads=n_heads)
model.to(device)
print(model)

prompt = 'hello thy'
prompt_tokens = [char2idx[c] for c in prompt]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_learning_rate(iteration):
    # Cosine decay with warmup. Think this is correct :) 
    # https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay/

    if iteration < warmup_iterations:
        return learning_rate * iteration/warmup_iterations
    if iteration > decay_iterations:
        return min_learning_rate

    ratio = (iteration - warmup_iterations)/(decay_iterations - warmup_iterations)
    cosine_decay = 0.5 * (1 + math.cos(math.pi*ratio))

    return (learning_rate - min_learning_rate)*cosine_decay + min_learning_rate

def test():
    model.eval()
    
    total_loss = 0.0
    counter = 0

    for i in range(200):
        x, y = get_batch('test')

        y_pred = model(x)
        y_pred = y_pred.view(-1, y_pred.size(-1))
        loss = criterion(y_pred, y.view(-1))
        total_loss += loss.item()
        counter += 1

    model.train()

    return total_loss/counter


counter = 0
best_loss = float('inf')

while counter < max_iterations:
    lr = get_learning_rate(counter)
        
    for param in optimizer.param_groups:
        param['lr'] = lr

    x, y = get_batch()
    optimizer.zero_grad()
    y_pred = model(x)
    y_pred = y_pred.view(-1, y_pred.size(-1))
    loss = criterion(y_pred, y.view(-1))

    print("IDX: {}, loss: {}, lr: {}".format(counter, loss.item(), lr))
    loss.backward()
    optimizer.step()

    if counter and counter%testing_interval== 0:
        checkpoint = model.state_dict()

        test_loss = test()
        print('test loss', test_loss)

        if test_loss < best_loss:
            print('Improved best loss. Saving checkpoint')
            best_loss = test_loss
            torch.save(checkpoint, 'checkpoint.pt')

        out = model.generate(prompt_tokens, 100, char2idx, idx2char)
        print(out)

    counter += 1
