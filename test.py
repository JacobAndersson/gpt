import fire
import pickle
from model import GPT, ModelConfig
import torch
import tiktoken
import dataclasses

tokenizer = tiktoken.get_encoding('gpt2')

def decode(tokens):
    return tokenizer.decode(tokens)

def encode(prompt):
    return tokenizer.encode_ordinary(prompt)

def main(path, device='cuda:0'):
    vocab_size = 50304
    embedding_dim = 768
    block_size = 1024
    n_layers = 24
    internal_dim = 3072
    n_heads = 12
    dropout = 0.0
    bias = False

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
    checkpoint = torch.load(path, map_location=device)

    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    while True:
        prompt = input('Prompt: ') 
        if prompt == 'q':
            break

        prompt_tokens = encode(prompt)

        out = model.generate(prompt_tokens, 25)
        print(out)
        print(decode(out))

if __name__ == '__main__':
    fire.Fire(main)
