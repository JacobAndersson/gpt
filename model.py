import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int
    block_size: int
    n_layers: int
    internal_dim: int
    n_heads: int
    dropout: int = 0.1
    device: str = 'cuda'

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        embedding_dim = config.embedding_dim
        block_size = config.block_size
        n_heads = config.n_heads

        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))

        #output projection
        self.w = nn.Linear(embedding_dim, embedding_dim)

        self.n_heads = n_heads
        self.dim = embedding_dim

    def forward(self, x):
        B, T, C = x.size()

        key = self.k(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        query = self.q(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        value = self.v(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)

        attention = (query @ key.transpose(-2, -1))*1/(math.sqrt(self.dim))
        attention = attention.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = attention@value
        y = attention.transpose(1,2).contiguous().view(B, T, C)

        y = self.w(y)

        return y
        
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        '''
        Components of decoder block
        1. self attention layer
        2. Multi Level Perptron (depth 2)
        3. layer normalization
        '''
        
        self.attention = SelfAttention(config)

        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.internal_dim),
            nn.GELU(),
            nn.Linear(config.internal_dim, config.embedding_dim),
            nn.Dropout(config.dropout)
        )

        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):

        x = x + self.attention(self.norm1(x))
        x = x = self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        '''
        Components of gpt
        token embedding
        learned positional embedding

        transformer block*n
        layer_norm
        '''

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.embedding_dim)

        self.layers = nn.Sequential(
            *(DecoderBlock(config) for _ in range(config.n_layers))
        )

        self.output_head = nn.Linear(config.embedding_dim, config.vocab_size)

        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        self.block_size = config.block_size
        self.config = config

    def forward(self, x):
        x_embed = self.token_embedding(x)
        pos = torch.arange(0, x.size()[1], device=self.config.device, dtype=torch.long).unsqueeze(0)
        pos_embed = self.pos_embedding(pos)
        x_rep = x_embed + pos_embed

        x = self.layers(x_rep)
        x = self.output_head(x)

        return x
    
    def generate(self, context, max_tokens):
        '''
        steps:
        convert context to idx tokens
        feed in to model
        get prediction of last token
        add to context and feed into network again
        '''

        context = torch.tensor(context, dtype=torch.long, device=self.config.device).unsqueeze(0)

        for i in range(max_tokens):
            idx = context[:, -1*self.block_size:]

            out = self.forward(idx)
            last_logit = out[:, -1, :]

            pred = F.softmax(last_logit, dim=1)
            pred_idx = torch.multinomial(pred, num_samples=1)

            context = torch.cat((context, pred_idx), dim=1)

            idx_list = context[0].cpu().tolist()
        
        return idx_list
