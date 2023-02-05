import numpy as np
import torch
import torch.nn as nn
from model import GPT, ModelConfig
import pickle
import math
import fire
from dataclasses import dataclass, asdict
import wandb
import tiktoken
import time
torch.manual_seed(42)

@dataclass
class BatchConfig:
    batch_size: int
    block_size: int
    device: str

def get_batch(data, config):
    rand = torch.randint(low=0, high=len(data) - config.block_size, size=(config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i: i+config.block_size]).astype(np.int64)) for i in rand])
    y = torch.stack([torch.from_numpy((data[i+1: i+config.block_size+1]).astype(np.int64)) for i in rand])

    x = x.to(config.device)
    y = y.to(config.device)

    return x,y

def test(model, test_idx, batch_config):
    with torch.no_grad():
        total_loss = 0.0
        counter = 0

        for i in range(200):
            x, y = get_batch(test_idx, batch_config)

            y_pred = model(x)
            y_pred = y_pred.view(-1, y_pred.size(-1))
            loss = nn.functional.cross_entropy(y_pred, y.view(-1))
            total_loss += loss.item()
            counter += 1

        return total_loss/counter

def main(
    device = 'cuda',
    batch_size = 64,
    micro_batch_steps = 1,
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
    use_wandb=False,
    dataset='shakespeare',
    lr_schedule = True,
    bias = False,
):
    internal_dim = 4*embedding_dim

    if dataset=='shakespeare':
        with open('meta.pkl', 'rb') as f:
            pkl = pickle.load(f)
            vocab_size = pkl['size']
            char2idx = pkl['cti']
            idx2char = pkl['itc']
    else:
        #encoded with tiktoken gpt2 tokenizer
        vocab_size=50257
        tokenizer = tiktoken.get_encoding('gpt2')

    def encode(text):
        if dataset=='shakespeare':
            return [char2idx[c] for c in text]
        return tokenizer.encode_ordinary(text)

    def decode(tokens):
        if dataset == 'shakespeare':
            return [idx2char[i] for i in tokens]
        return tokenizer.decode(tokens)

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
        device,
        bias,
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
    num_params = model.num_params()
    print('num parameter {}M'.format(round(num_params/1_000_000, 2)))
    model.to(device)
    print(model)

    if use_wandb:
        print('Using wandb')
        wandb.init(config={**asdict(config), **asdict(batch_config)})

    prompt = 'hello thy'
    prompt_tokens = encode(prompt)
    
    criterion = nn.CrossEntropyLoss()

    #betas, eps and weightdecay from Cramming paper
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-12)

    counter = 0
    best_loss = float('inf')

    while counter < max_iterations:
        if lr_schedule:
            lr = get_learning_rate(counter)
        else:
            lr = learning_rate
                
        for param in optimizer.param_groups:
            param['lr'] = lr

        t0 = time.time()

        for _ in range(micro_batch_steps):
            x, y = get_batch(train_idx, batch_config)
            y_pred = model(x)
            y_pred = y_pred.view(-1, y_pred.size(-1))
            loss = criterion(y_pred, y.view(-1))
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        elapsed = round((t1-t0), 3)

        print("IDX: {}, loss: {}, lr: {}, batch took: {}s".format(counter, loss.item(), lr, elapsed))
        if use_wandb:
            wandb.log({ 'loss': loss.item(), "lr": lr })


        if counter and counter%testing_interval== 0:
            model.eval()
            test_loss = test(model, val_idx, batch_config)
            model.train()
            print('Test loss', test_loss)
            if use_wandb:
                wandb.log({ "test_loss": test_loss })

            if test_loss < best_loss:
                print('Improved best loss. Saving checkpoint')
                checkpoint = {
                    "model": model.state_dict(),
                    "lr": learning_rate,
                    "iter": counter,
                }
                best_loss = test_loss
                torch.save(checkpoint, 'checkpoint.pt')

            out_tokens = model.generate(prompt_tokens, 100)
            out = decode(out_tokens)
            print(out)

        counter += 1

if __name__ == '__main__':
    fire.Fire(main)
