import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)


class BigramLM(nn.Module):
    def __init__(self, vocab_sz, emb_sz):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_sz)

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
