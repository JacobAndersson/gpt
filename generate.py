import model 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
block_size = 64
num_epochs = 16
n_layers = 12
embedding_dim = 768
internal_dim = 4*embedding_dim
n_heads = 4


model = model.GPT(vocab_size=vocab_size, embedding_dim=embedding_dim, block_size=block_size, n_layers=n_layers, internal_dim=internal_dim, n_heads=n_heads)
model.to(device)

model.sample('hell')
