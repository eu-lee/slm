import numpy
import torch.nn as nn
from torch.nn import functional as F


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        '''
        conversion between tokens and predicted next token.
        it is a vocab_size x vocab_size matrix, representing the raw score. Given a letter i, the probability of j occuring
        is represented by token_embedding_table[i][j] (not normalized)
        '''
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        '''
        returned in form (B, T, C)
        given a set of tokens, idx, the logit distribution (raw) are given by the C channel
        '''
        logits = self.token_embedding_table(idx)

        B,T,C = logits.shape

        # change dimensionality to fit crossEntropy
        logits = logits.view(B*T,C) 
        '''
        now, given targets, which are the identities of the next tokens, how good were our guesses from logits?
        '''

        loss = F.cross_entropy(logits,targets)
        return logits, loss
