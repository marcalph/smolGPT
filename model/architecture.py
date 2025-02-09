import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

#hparams
batch_sz  = 32
context_sz = 8
max_steps = 20000
eval_interval = 1000
lr = 3e-4
device = torch.device("cpu")#torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
eval_steps = 200
d_embd = 32
vocab_sz = 65 
d_head=8


class Head(nn.Module):
    """ 1 single head of attention"""
    def __init__(self, d_embd, d_head):
        super().__init__()
        self.key = nn.Linear(d_embd, d_head, bias = False)
        self.query = nn.Linear(d_embd, d_head, bias = False)
        self.value = nn.Linear(d_embd, d_head, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(context_sz, context_sz)))        

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute affinity score
        wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # weigthed sum of values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out



class BigramLMv2(nn.Module):
    def __init__(self, vocab_sz, d_embd, d_head):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d_embd)
        self.position_embeddings = nn.Embedding(context_sz, d_embd)
        self.sa_head = Head(d_embd, d_head)
        self.lm_head = nn.Linear(d_head, vocab_sz)

    def forward(self, ix, targets=None):
        B, T = ix.shape

        # idx and targets are (B, T) shaped tensors
        tok_emb = self.emb(ix) # (B, T, C)
        pos_emb = self.position_embeddings(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_head(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_sz)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.size()
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 
        return logits, loss
    
    def generate(self, idx, max_tokens):
        # idx is a (B, T) shaped tensor
        for _ in range(max_tokens):
            #crop idx the the last context_sz tokens
            idx_cond = idx[:, -context_sz:] # (B, context_sz)
            logits, loss = self(idx_cond)
            # focus only on the last time step
            last_logits = logits[:, -1, :] # (B, C)
            # apply softmax
            probs = F.softmax(last_logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) 
        return idx



class BigramLM(nn.Module):
    def __init__(self, vocab_sz, d_embd):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, vocab_sz)

    def forward(self, ix, targets=None):
        # idx and targets are (B, T) shaped tensors
        logits = self.emb(ix) # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.size()
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 
        return logits, loss
    
    def generate(self, idx, max_tokens):
        # idx is a (B, T) shaped tensor
        for _ in range(max_tokens):
            logits, loss = self(idx)
            # focus only on the last time step
            last_logits = logits[:, -1, :] # (B, C)
            # apply softmax
            probs = F.softmax(last_logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
