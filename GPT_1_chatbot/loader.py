import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

# find checkpoint
CKPT_CANDIDATES = [
    './checkpoints/final_model_checkpoint.pth',
    './checkpoints/checkpoint_iter_100.pth',
    './checkpoints/checkpoint_iter_0.pth',
    './model_checkpoint.pth'
]
ckpt_path = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
if ckpt_path is None:
    raise FileNotFoundError('No checkpoint found. Expected one of: ' + ', '.join(CKPT_CANDIDATES))

# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load checkpoint
ckpt = torch.load(ckpt_path, map_location=device)

# load tokenizer from trained tokenizer directory (preferred) or from checkpoint metadata
if os.path.isdir('./tokenizer'):
    tokenizer = GPT2TokenizerFast.from_pretrained('./tokenizer')
else:
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# restore hyperparameters from checkpoint
vocab_size = len(tokenizer)
n_embed = ckpt.get('n_embed')
n_layers = ckpt.get('n_layers')
n_head = ckpt.get('n_head')
dropout = ckpt.get('dropout', 0.0)
block_size = ckpt.get('block_size', 384)

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

class GPT(nn.Module):
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

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, min_new_tokens=3):
        eos_token_id = tokenizer.eos_token_id
        user_token_id = tokenizer.get_vocab().get("<|user|>")
        assistant_token_id = tokenizer.get_vocab().get("<|assistant|>")
        tokens_generated = 0

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # apply temperature
            logits = logits / max(1e-9, temperature)

            # apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                logits = logits_filtered

            # prevent immediate EOS or <|user|> until minimum tokens generated
            if tokens_generated < min_new_tokens:
                if eos_token_id is not None:
                    logits[:, eos_token_id] = float('-inf')
                if user_token_id is not None:
                    logits[:, user_token_id] = float('-inf')
                if assistant_token_id is not None:
                    logits[:, assistant_token_id] = float('-inf')

            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)

            next_token = idx_next.item()
            # stop if EOS token or <|user|> token generated after min_new_tokens
            if tokens_generated >= min_new_tokens and (
                (eos_token_id is not None and next_token == eos_token_id) or
                (user_token_id is not None and next_token == user_token_id) or
                (assistant_token_id is not None and next_token == assistant_token_id)
            ):
                break

            idx = torch.cat((idx, idx_next), dim=1)
            tokens_generated += 1
        return idx

# helper encode/decode using tokenizer
def encode(text):
    return tokenizer.encode(text, add_special_tokens=False)

def decode(token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# build model and load weights
model = GPT(vocab_size)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()


total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

print('Loaded model from:', ckpt_path)
print('Device:', device)

def chat(prompt, history=None, max_new_tokens=60, temperature=0.7, top_k=40, min_new_tokens=3, history_max_tokens=None):
    """Return generated reply and updated history.
    Args:
        prompt: Input text from the user.
        history: List of (role, text) tuples to include in context.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        min_new_tokens: Minimum tokens to generate before allowing EOS/user stop.
        history_max_tokens: Maximum allowed tokens for historical chat; if exceeded, history is cleared.
    Returns:
        reply (str), updated_history (list)
    """
    if history is None:
        history = []

    # default history max tokens to model block size minus a safety margin
    if history_max_tokens is None:
        history_max_tokens = max(1, block_size - 50)

    # compute token count of the historical chat
    hist_token_count = 0
    for role, text in history:
        if role == "user":
            hist_token_count += len(tokenizer.encode("<|user|> " + text + " <|assistant|>", add_special_tokens=False))
        else:
            hist_token_count += len(tokenizer.encode("<|assistant|> " + text + " <|eos|>", add_special_tokens=False))

    # if history exceeds allowed tokens, clear it and notify
    if hist_token_count > history_max_tokens:
        print(f"History exceeded {history_max_tokens} tokens (was {hist_token_count}); clearing history.")
        history = []

    # build prompt tokens including recent history
    toks = []
    for role, text in history:
        if role == "user":
            toks += tokenizer.encode("<|user|> " + text + " <|assistant|>", add_special_tokens=False)
        else:
            toks += tokenizer.encode("<|assistant|> " + text + " <|eos|>", add_special_tokens=False)

    # add current user prompt
    toks += tokenizer.encode("<|user|> " + prompt + " <|assistant|>", add_special_tokens=False)

    idx = torch.tensor([toks], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, min_new_tokens=min_new_tokens)
    out_tokens = out[0, idx.shape[1]:].tolist()

    # if reply is empty, fallback to greedy decoding
    if len(out_tokens) == 0:
        idx = torch.tensor([toks], dtype=torch.long).to(device)
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -block_size:])
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
        out_tokens = idx[0, len(toks):].tolist()

    reply = tokenizer.decode(out_tokens, skip_special_tokens=True)
    history.append(("assistant", reply))
    return reply, history


if __name__ == '__main__':
    try:
        history = []
        while True:
            prompt = input('You: ')
            if prompt.strip().lower() in ('quit', 'exit'):
                print('Exiting.')
                break
            reply, history = chat(prompt, history=history)
            print('Bot:', reply)
    except KeyboardInterrupt:
        print('\nInterrupted. Exiting.')
