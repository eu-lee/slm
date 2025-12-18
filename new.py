import numpy
import os
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

#hyperparams
block_size = 8 #context window
batch_size = 32 #parallel operations

epochs = 10000 #training iterations

device = torch.device("mps")
learning_rate = 1e-3

# how often to run evaluation
eval_interval = 100
n_embed = 32

#the average the loss is taken by
eval_iters = 200

# < data loading >

path = os.path.join("data", "shakespeare.txt")
with open(path,"r", encoding= "UTF-8") as f:
    text = f.read()

'''
tokenization by character
'''
chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text),dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]


'''
essentially, we are given an index of [1,8] tokens, the expected output for the
i-th token is j in [2,9]

x = train_data[:block_size] # input tokens
y = train_data[1:block_size+1] # expected output logits
'''
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

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        '''
        conversion between tokens and predicted next token.
        it is a vocab_size x vocab_size matrix, representing the raw score. Given a letter i, the probability of j occuring
        is represented by token_embedding_table[i][j] (not normalized)
        '''
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)

    def forward(self, idx, targets = None):

        B,T = idx.shape
        token_embedding = self.token_embedding_table(idx) # (B,T,C)

        pos_embedding = self.position_embedding_table(torch.arange(T,device = device))
        
        x = token_embedding + pos_embedding #enable position-based embeddings 

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
            logits, loss = self(idx)
            logits = logits[ :, -1,:] #most recent timestamp, hence "bi-gram"
            probabilities = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probabilities,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)

        return idx


model = BigramModel(vocab_size)

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