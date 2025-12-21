import torch
import torch.nn as nn
import torch.nn.functional as F

# Load checkpoint (saved by train.py)
ckpt_path = 'model_checkpoint.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# restore token maps and hyperparameters
stoi = ckpt['stoi']
itos = ckpt['itos']
vocab_size = ckpt['vocab_size']
n_embed = ckpt['n_embed']
n_layers = ckpt['n_layers']
n_head = ckpt['n_head']
dropout = ckpt['dropout']
block_size = ckpt['block_size']

# define model classes (matches train.py)
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
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
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
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_embedding = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embedding + pos_embedding
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# helper encode/decode
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[int(i)] for i in l])

# build model and load weights
model = BigramModel(vocab_size)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

print("Loaded model from:", ckpt_path)
print("Device:", device)

# simple interactive loop
if __name__ == '__main__':
    try:
        while True:
            prompt = input('You: ')
            if prompt.strip().lower() in ('quit', 'exit'):
                print('Exiting.')
                break
            encoded = encode(prompt)
            if len(encoded) == 0:
                encoded = [0]
            idx = torch.tensor([encoded], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=200)
            out_list = out[0].tolist()
            # show only generated part after the prompt
            generated = decode(out_list)[len(prompt):]
            print('Bot:', generated)
    except KeyboardInterrupt:
        print('\nInterrupted. Exiting.')
