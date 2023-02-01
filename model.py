import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SelfAttention(nn.Module):
    def __init__(self, block_size, embedding_dim, n_heads):
        super().__init__()

        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))

        #output projection
        self.w = nn.Linear(embedding_dim, embedding_dim )

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
    def __init__(self, block_size, embedding_dim, internal_dim, n_heads, dropout):
        super().__init__()
        '''
        Components of decoder block
        1. self attention layer
        2. Multi Level Perptron (depth 2)
        3. layer normalization
        '''
        
        self.attention = SelfAttention(block_size, embedding_dim, n_heads)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, embedding_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):

        x = x + self.attention(self.norm1(x))
        x = x = self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, n_layers, internal_dim, n_heads, dropout=0.1):
        super().__init__()
        '''
        Components of gpt
        token embedding
        learned positional embedding

        transformer block*n
        layer_norm
        '''

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(block_size, embedding_dim)

        self.layers = nn.Sequential(
            *(DecoderBlock(block_size, embedding_dim, internal_dim, n_heads, dropout) for _ in range(n_layers))
        )

        self.output_head = nn.Linear(embedding_dim, vocab_size)

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.block_size = block_size

    def forward(self, x):
        x_embed = self.token_embedding(x)
        pos = torch.arange(0, x.size()[1], device=device, dtype=torch.long).unsqueeze(0)
        pos_embed = self.pos_embedding(pos)
        x_rep = x_embed + pos_embed

        x = self.layers(x_rep)
        x = self.output_head(x)

        return x
    
    def generate(self, context, max_tokens, char2idx, idx2char):
        '''
        steps:
        convert context to idx tokens
        feed in to model
        get prediction of last token
        add to context and feed into network again
        '''

        context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        for i in range(max_tokens):
            idx = context[:, -1*self.block_size:]

            out = self.forward(idx)
            last_logit = out[:, -1, :]

            pred = F.softmax(last_logit, dim=1)
            pred_idx = torch.multinomial(pred, num_samples=1)

            context = torch.cat((context, pred_idx), dim=1)

            idx_list = context[0].cpu().tolist()
        
        out = "".join([idx2char[c] for c in idx_list])
        return out