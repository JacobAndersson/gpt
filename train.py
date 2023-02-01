import numpy as np
import torch
import torch.nn as nn
from model import GPT, ModelConfig
import pickle
import math
import fire
from dataclasses import dataclass

@dataclass
class BatchConfig:
    batch_size: int
    block_size: int
    device: str

def get_batch(data, config):
    rand = torch.randint(low=0, high=len(data) - config.block_size, size=(config.batch_size,))

    x = torch.stack([torch.from_numpy((data[x: x+config.block_size]).astype(np.int64)) for x in rand])
    y = torch.stack([torch.from_numpy((data[x+1: x+config.block_size+1]).astype(np.int64)) for x in rand])

    x = x.to(config.device)
    y = y.to(config.device)

    return x,y

def test(model, test_idx, batch_config):
    
    total_loss = 0.0
    counter = 0

    for i in range(200):
        x, y = get_batch(test_idx, batch_config)

        y_pred = model(x)
        y_pred = y_pred.view(-1, y_pred.size(-1))
        loss = criterion(y_pred, y.view(-1))
        total_loss += loss.item()
        counter += 1

    return total_loss/counter

def main(
    device = 'cuda',
    batch_size = 64,
    block_size = 64,
    num_epochs = 16,
    n_layers = 10,
    embedding_dim = 768,
    n_heads = 8,
    learning_rate = 6e-5,
    min_learning_rate = 6e-6,
    warmup_iterations = 2000,
    decay_iterations = 60000,
    max_iterations = 60000,
    testing_interval = 1000,
    dropout = 0.1,
):
    internal_dim = 4*embedding_dim

    with open('meta.pkl', 'rb') as f:
        pkl = pickle.load(f)
        vocab_size = pkl['size']
        char2idx = pkl['cti']
        idx2char = pkl['itc']


    train_idx = np.memmap('train.bin', dtype=np.uint16)
    val_idx = np.memmap('test.bin', dtype=np.uint16)

    config = ModelConfig(
        vocab_size,
        embedding_dim,
        block_size,
        n_layers,
        internal_dim,
        n_heads,
        dropout,
        device
    )
    print(config)

    batch_config = BatchConfig(batch_size, block_size, device)
    print(batch_config)

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


    model = GPT(config)
    model.to(device)
    print(model)

    prompt = 'hello thy'
    prompt_tokens = [char2idx[c] for c in prompt]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    counter = 0
    best_loss = float('inf')

    while counter < max_iterations:
        lr = get_learning_rate(counter)
            
        for param in optimizer.param_groups:
            param['lr'] = lr

        x, y = get_batch(train_idx, batch_config)
        optimizer.zero_grad()
        y_pred = model(x)
        y_pred = y_pred.view(-1, y_pred.size(-1))
        loss = criterion(y_pred, y.view(-1))

        print("IDX: {}, loss: {}, lr: {}".format(counter, loss.item(), lr))
        loss.backward()
        optimizer.step()

        if counter and counter%testing_interval== 0:
            checkpoint = model.state_dict()

            model.eval()
            test_loss = test(model, test_idx, batch_config)
            model.train()
            print('test loss', test_loss)

            if test_loss < best_loss:
                print('Improved best loss. Saving checkpoint')
                best_loss = test_loss
                torch.save(checkpoint, 'checkpoint.pt')

            out = model.generate(prompt_tokens, 100, char2idx, idx2char)
            print(out)

        counter += 1

if __name__ == '__main__':
    fire.Fire(main)
