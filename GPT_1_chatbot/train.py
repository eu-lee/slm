import numpy
import torch
import json
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel

import numpy
import os
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

#hyperparams
block_size = 384 #context window
batch_size = 48 #parallel operations

n_embed = 512 #embedding dimension
n_layers = 6
n_head = 8

epochs = 5000 #training iterations
# how often to run evaluation
eval_interval = 200
#the average the loss is taken by
eval_iters = 200

dropout = 0.15
learning_rate = 3e-4


device = torch.device("cuda")

# < data loading >

# Load the trained tokenizer
print("Loading trained tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("./tokenizer")
vocab_size = tokenizer.vocab_size
print(f"Tokenizer loaded with vocab size: {vocab_size}")

# Load pre-tokenized data
print("Loading pre-tokenized data...")
token_ids_array = numpy.load("./tokenizer/processed_data/token_ids.npy", allow_pickle=True)

# Load metadata
with open("./tokenizer/processed_data/metadata.json") as f:
    metadata = json.load(f)

block_size = metadata['block_size']
print(f"Block size from metadata: {block_size}")

# Flatten token arrays into a single tensor
data = torch.tensor(numpy.concatenate([numpy.array(tokens, dtype=numpy.int32) for tokens in token_ids_array]), dtype=torch.long)
print(f"Total tokens loaded: {len(data)}")

# Define decode function for generation
decode = lambda l: tokenizer.decode(l, skip_special_tokens=False)

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb,yb = get_batch("train")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B,T,T)

        wei = self.dropout(wei)
        v = self.value(x) # (B,T,head_size)

        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.proj(torch.cat([h(x) for h in self.heads], dim=-1))
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4* n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# transformer block

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x+ self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)

    def forward(self, idx, targets = None):

        B,T = idx.shape
        token_embedding = self.token_embedding_table(idx) # (B,T,C)

        pos_embedding = self.position_embedding_table(torch.arange(T,device = device))
        
        x = token_embedding + pos_embedding #enable position-based embeddings 
        x = self.blocks(x) # (B,T,C)
        x = self.ln_final(x) # (B,T,C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # holds token identities and positions

        if targets is None:
            loss = None

        else:
            B,T,C = logits.shape

            # change dimensionality to fit crossEntropy
            logits = logits.view(B*T,C) 
            targets = targets.view(B*T)
            '''
            now, given targets, which are the identities of the next tokens, how good were our guesses from logits?
            '''
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # use only the recent context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # last time step
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPT(len(tokenizer))  # Use actual tokenizer size including special tokens

model.to(device)
'''
#perform a forward pass
logits, loss = model(xb,yb)

print(loss)

# zeros of 1,1 is like the newline char \n
print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
'''


#< training loop>
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

#batch_size = 32

train_losses = []
val_losses = []
eval_steps = []

for iter in range(epochs):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == epochs - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        eval_steps.append(iter)
        # save a checkpoint so training progress is kept
        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'n_embed': n_embed,
                'n_layers': n_layers,
                'n_head': n_head,
                'dropout': dropout,
                'block_size': block_size,
                'iter': iter,
                'tokenizer_name': 'GPT2TokenizerFast',
            }
            torch.save(checkpoint, f'./checkpoints/checkpoint_iter_{iter}.pth')
            print(f"Saved checkpoint at iter {iter} -> model_checkpoint.pth")
        except Exception as e:
            print("Warning: failed to save checkpoint:", e)

    # sample a batch of data
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device = device), max_new_tokens=500)[0].tolist()))

# final checkpoint save
try:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'n_embed': n_embed,
        'n_layers': n_layers,
        'n_head': n_head,
        'dropout': dropout,
        'block_size': block_size,
        'iter': epochs - 1,
        'tokenizer_name': 'GPT2TokenizerFast',
    }
    torch.save(checkpoint, './checkpoints/final_model_checkpoint.pth')
    print("Saved final checkpoint -> model_checkpoint.pth")
except Exception as e:
    print("Warning: failed to save final checkpoint:", e)

# Plot training and validation loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(eval_steps, train_losses, label='Training Loss', marker='o')
plt.plot(eval_steps, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()