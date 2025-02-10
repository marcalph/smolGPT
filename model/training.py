from dataclasses import dataclass
import torch

BATCH_SZ = 8
CONTEXT_SZ = 16


@dataclass
class Splitter:
    train_sz: float = 0.9

    def sequential_split(self, data):
        n = int(len(data) * self.train_sz)
        return {"train": data[:n], "val": data[n:]}


def make_batches(tokenized_data, device, batch_sz=BATCH_SZ, context_sz=CONTEXT_SZ):
    for _ in range(len(tokenized_data) // batch_sz):
        ix = torch.randint(0, len(tokenized_data) - context_sz, (batch_sz,))
        x = torch.stack([tokenized_data[i : i + context_sz] for i in ix])
        y = torch.stack([tokenized_data[i + 1 : i + context_sz + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        yield x, y


@torch.no_grad()
def estimate_loss(model, data, eval_steps, batch_sz, context_sz):
    model.eval()
    losses = torch.zeros(eval_steps)
    for k in range(eval_steps):
        xb, yb = next(
            make_batches(data, next(model.parameters()).device, batch_sz, context_sz)
        )
        _, loss = model(xb, yb)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()
