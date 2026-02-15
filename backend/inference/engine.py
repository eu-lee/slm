from __future__ import annotations

import os
import threading
import asyncio
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast


# Resolve paths relative to the model directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "model"
CKPT_PATH = BASE_DIR / "checkpoints" / "final_model_checkpoint.pth"
TOKENIZER_DIR = BASE_DIR / "tokenizer"


class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.proj(torch.cat([h(x) for h in self.heads], dim=-1))
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, n_head, block_size, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=self.device))
        x = token_embedding + pos_embedding
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


class ModelEngine:
    """Wraps the GPT model for inference with streaming token generation."""

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self._lock = threading.Lock()

        # Load checkpoint
        if not CKPT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")
        ckpt = torch.load(str(CKPT_PATH), map_location="cpu")

        # Load tokenizer
        if TOKENIZER_DIR.is_dir():
            self.tokenizer = GPT2TokenizerFast.from_pretrained(str(TOKENIZER_DIR))
        else:
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        # Hyperparameters from checkpoint
        vocab_size = len(self.tokenizer)
        n_embed = ckpt.get('n_embed')
        n_layers = ckpt.get('n_layers')
        n_head = ckpt.get('n_head')
        dropout = ckpt.get('dropout', 0.0)
        self.block_size = ckpt.get('block_size', 384)

        # Build and load model
        self.model = GPT(vocab_size, n_embed, n_layers, n_head, self.block_size, dropout, self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Special token IDs
        vocab = self.tokenizer.get_vocab()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.user_token_id = vocab.get("<|user|>")
        self.assistant_token_id = vocab.get("<|assistant|>")

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {total_params:,} parameters on {self.device}")

    def _build_prompt_tokens(self, message: str, history: list[dict]) -> tuple[list[int], bool]:
        """Build token sequence from conversation history + new message.

        history: list of {"role": "user"|"assistant", "content": "..."}
        Returns (tokens, history_cleared) - history_cleared is True if context was exceeded.
        """
        history_max_tokens = max(1, self.block_size - 50)
        history_cleared = False

        # Compute history token count
        hist_toks = []
        for msg in history:
            if msg["role"] == "user":
                hist_toks += self.tokenizer.encode(
                    f"<|user|> {msg['content']} <|assistant|>", add_special_tokens=False
                )
            else:
                hist_toks += self.tokenizer.encode(
                    f"<|assistant|> {msg['content']} <|eos|>", add_special_tokens=False
                )

        if len(hist_toks) > history_max_tokens:
            print(f"History exceeded {history_max_tokens} tokens (was {len(hist_toks)}); clearing.")
            hist_toks = []
            history_cleared = True

        # Add current user message
        toks = hist_toks + self.tokenizer.encode(
            f"<|user|> {message} <|assistant|>", add_special_tokens=False
        )
        return toks, history_cleared

    def _generate_tokens(self, toks: list[int], max_new_tokens: int, temperature: float,
                         top_k: int, min_new_tokens: int, token_queue: asyncio.Queue, loop):
        """Synchronous generation that pushes tokens to an async queue."""
        with self._lock:
            idx = torch.tensor([toks], dtype=torch.long).to(self.device)
            tokens_generated = 0

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -self.block_size:]
                    logits, _ = self.model(idx_cond)
                    logits = logits[:, -1, :]

                    logits = logits / max(1e-9, temperature)

                    if top_k is not None:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=1)
                        logits_filtered = torch.full_like(logits, float('-inf'))
                        logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                        logits = logits_filtered

                    if tokens_generated < min_new_tokens:
                        if self.eos_token_id is not None:
                            logits[:, self.eos_token_id] = float('-inf')
                        if self.user_token_id is not None:
                            logits[:, self.user_token_id] = float('-inf')
                        if self.assistant_token_id is not None:
                            logits[:, self.assistant_token_id] = float('-inf')

                    probs = F.softmax(logits, dim=1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    next_token = idx_next.item()

                    if tokens_generated >= min_new_tokens and (
                        (self.eos_token_id is not None and next_token == self.eos_token_id) or
                        (self.user_token_id is not None and next_token == self.user_token_id) or
                        (self.assistant_token_id is not None and next_token == self.assistant_token_id)
                    ):
                        break

                    idx = torch.cat((idx, idx_next), dim=1)
                    tokens_generated += 1

                    token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                    loop.call_soon_threadsafe(token_queue.put_nowait, token_text)

            # Signal completion
            loop.call_soon_threadsafe(token_queue.put_nowait, None)

    async def generate_stream(self, message: str, history: Optional[list[dict]] = None,
                              max_new_tokens: int = 60, temperature: float = 0.7,
                              top_k: int = 40, min_new_tokens: int = 3):
        """Async generator that yields token strings one at a time.

        May yield a special sentinel dict {"context_cleared": True} before tokens
        if the history exceeded the context window.
        """
        if history is None:
            history = []

        toks, history_cleared = self._build_prompt_tokens(message, history)
        if history_cleared:
            yield {"context_cleared": True}
        token_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        # Run synchronous generation in a thread
        asyncio.get_event_loop().run_in_executor(
            None,
            self._generate_tokens,
            toks, max_new_tokens, temperature, top_k, min_new_tokens, token_queue, loop,
        )

        # Yield tokens as they arrive
        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield token
